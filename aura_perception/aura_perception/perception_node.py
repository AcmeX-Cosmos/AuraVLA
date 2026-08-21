#!/usr/bin/env python3
"""
AuraVLA Perception Node

ROS2 node for scene understanding and doability evaluation.
"""

import rclpy
from rclpy.node import Node
from aura_interfaces.srv import EvaluateDoable
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import yaml
from pathlib import Path

from aura_perception.vlm_client import NvidiaVLMClient, VLMConfig
from aura_perception.doable_evaluator import DoableEvaluator
from aura_perception.scene_names import SceneNameResolver


class PerceptionNode(Node):
    """
    ROS2 node for perception and doability evaluation
    """

    def __init__(self):
        super().__init__('aura_perception_node')

        # Load configuration
        self.declare_parameter('config_file', '')
        config_file = self.get_parameter('config_file').value

        if config_file:
            config = self._load_config(config_file)
        else:
            config = self._get_default_config()

        # Initialize VLM client
        vlm_config = VLMConfig(
            model=config['nvidia']['model'],
            base_url=config['nvidia']['base_url'],
            api_key=config['nvidia']['api_key'],
            max_tokens=config['nvidia']['max_tokens'],
            temperature=config['nvidia']['temperature'],
            top_p=config['nvidia']['top_p'],
            request_timeout_sec=config['nvidia']['request_timeout_sec'],
            max_retries=config['nvidia']['max_retries'],
            image_max_edge=config['agent']['image_max_edge']
        )

        vlm_client = NvidiaVLMClient(vlm_config)

        # Initialize doable evaluator
        self.evaluator = DoableEvaluator(vlm_client)

        # Initialize scene name resolver
        scene_names = config.get('scene', {}).get('canonical_names', {})
        self.name_resolver = SceneNameResolver(scene_names)

        # CV Bridge for image conversion
        self.bridge = CvBridge()

        # Create service
        self.doable_service = self.create_service(
            EvaluateDoable,
            'aura/perception/doable',
            self.evaluate_doable_callback
        )

        self.get_logger().info('AuraVLA Perception Node initialized')

    def evaluate_doable_callback(self, request, response):
        """
        Service callback for doability evaluation

        Args:
            request: EvaluateDoable request
            response: EvaluateDoable response

        Returns:
            response
        """
        self.get_logger().info(f'Evaluating doability: {request.request.instruction}')

        try:
            # Convert ROS images to numpy
            rgb_image = None
            if request.request.rgb_image.data:
                rgb_image = self.bridge.imgmsg_to_cv2(
                    request.request.rgb_image,
                    desired_encoding='rgb8'
                )

            # Build context
            context = {}
            if request.request.previous_error:
                context['previous_error'] = request.request.previous_error

            # Evaluate doability
            result = self.evaluator.evaluate(
                instruction=request.request.instruction,
                rgb_image=rgb_image,
                context=context
            )

            # Fill response
            response.doable = result.get('doable', False)
            response.confidence = float(result.get('confidence', 0.0))
            response.reason = result.get('reason', '')
            response.required_objects = result.get('required_objects', [])

            self.get_logger().info(
                f"Result: doable={response.doable}, "
                f"confidence={response.confidence:.2f}"
            )

        except Exception as e:
            self.get_logger().error(f'Doable evaluation failed: {e}')
            response.doable = False
            response.confidence = 0.0
            response.reason = f'Evaluation error: {str(e)}'

        return response

    def _load_config(self, config_file: str) -> dict:
        """Load configuration from YAML file"""
        with open(config_file, 'r') as f:
            return yaml.safe_load(f)

    def _get_default_config(self) -> dict:
        """Get default configuration"""
        return {
            'nvidia': {
                'model': 'nvidia/nemotron-nano-12b-v2-vl',
                'base_url': 'https://integrate.api.nvidia.com/v1',
                'api_key': '',
                'max_tokens': 768,
                'temperature': 0.2,
                'top_p': 0.9,
                'request_timeout_sec': 300.0,
                'max_retries': 1
            },
            'agent': {
                'image_max_edge': 448,
                'max_history_messages': 2
            },
            'scene': {
                'canonical_names': {}
            }
        }


def main(args=None):
    """Main entry point"""
    rclpy.init(args=args)
    node = PerceptionNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
