"""
Doable Evaluator

Evaluates whether a task is feasible given the current scene.
"""

from typing import Dict, Any, Optional
from .vlm_client import VLMClient


class DoableEvaluator:
    """
    Evaluates task doability using VLM
    """

    def __init__(self, vlm_client: VLMClient):
        """
        Initialize evaluator

        Args:
            vlm_client: VLM client for inference
        """
        self.vlm_client = vlm_client

    def evaluate(
        self,
        instruction: str,
        rgb_image=None,
        depth_image=None,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Evaluate if task is doable

        Args:
            instruction: Natural language instruction
            rgb_image: RGB image of scene
            depth_image: Optional depth image
            context: Optional context from previous attempts

        Returns:
            Dict with doable, confidence, reason
        """
        prompt = self._build_doable_prompt(instruction, context)

        try:
            result = self.vlm_client.infer(prompt, rgb_image)

            # Ensure required fields
            if 'doable' not in result:
                result['doable'] = False
            if 'confidence' not in result:
                result['confidence'] = 0.0
            if 'reason' not in result:
                result['reason'] = 'Unable to evaluate'

            return result

        except Exception as e:
            return {
                'doable': False,
                'confidence': 0.0,
                'reason': f'Evaluation error: {str(e)}'
            }

    def _build_doable_prompt(
        self,
        instruction: str,
        context: Optional[Dict[str, Any]]
    ) -> str:
        """Build prompt for doability evaluation"""

        prompt = f"""Analyze the scene and determine if the following task is doable.

Task: {instruction}

Consider:
1. Are all required objects visible in the scene?
2. Are objects in reachable positions?
3. Are there any physical constraints preventing execution?
"""

        if context and 'previous_error' in context:
            prompt += f"\n\nNote: Previous attempt failed with: {context['previous_error']}"

        prompt += """

Respond in JSON format:
{
    "doable": true/false,
    "confidence": 0.0-1.0,
    "reason": "Brief explanation",
    "required_objects": ["list", "of", "objects"]
}
"""
        return prompt
