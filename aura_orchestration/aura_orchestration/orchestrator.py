"""
Main Orchestrator

Coordinates the complete closed-loop control:
Perception → Planning → Execution → Verification → Replan

AuraVLA closed-loop orchestration implementation.
"""

from typing import Dict, Any, Optional
from enum import Enum


class ExecutionState(Enum):
    """Task execution states"""
    IDLE = "idle"
    EVALUATING = "evaluating"
    PLANNING = "planning"
    EXECUTING = "executing"
    CHECKING = "checking"
    COMPLETED = "completed"
    FAILED = "failed"
    REPLANNING = "replanning"


class Orchestrator:
    """
    Main orchestrator for closed-loop control

    Coordinates: Doable → Plan → Execute → Check → Replan
    """

    def __init__(self, max_replans: int = 2):
        """
        Initialize orchestrator

        Args:
            max_replans: Maximum replanning attempts
        """
        self.max_replans = max_replans
        self.state = ExecutionState.IDLE

    def execute_task(
        self,
        instruction: str,
        scene_data: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """
        Execute task with closed-loop control

        Args:
            instruction: Natural language instruction
            scene_data: Optional scene information

        Returns:
            Execution result
        """
        attempts = []
        replan_context = {}

        for attempt in range(self.max_replans + 1):
            print(f"\n--- Attempt {attempt + 1}/{self.max_replans + 1} ---")

            try:
                # 1. Doable evaluation
                self.state = ExecutionState.EVALUATING
                doable_result = self._evaluate_doable(
                    instruction,
                    scene_data,
                    replan_context
                )

                if not doable_result.get('doable', False):
                    return {
                        'success': False,
                        'state': 'not_doable',
                        'reason': doable_result.get('reason', 'Task not doable'),
                        'attempts': attempts
                    }

                # 2. Planning
                self.state = ExecutionState.PLANNING
                plan = self._generate_plan(instruction, replan_context)

                # 3. Execution
                self.state = ExecutionState.EXECUTING
                exec_result = self._execute_plan(plan)

                if not exec_result.get('success', False):
                    attempts.append({
                        'attempt': attempt + 1,
                        'state': 'execution_failed',
                        'error': exec_result.get('message', '')
                    })
                    replan_context = {
                        'execution_error': exec_result.get('message', '')
                    }
                    continue

                # 4. Verification
                self.state = ExecutionState.CHECKING
                check_result = self._check_completion(plan, exec_result)

                attempts.append({
                    'attempt': attempt + 1,
                    'state': 'completed',
                    'check': check_result
                })

                if check_result.get('success', False):
                    self.state = ExecutionState.COMPLETED
                    return {
                        'success': True,
                        'state': 'completed',
                        'reason': check_result.get('reason', ''),
                        'attempts': attempts
                    }

                # Need replan?
                if not check_result.get('need_replan', False):
                    self.state = ExecutionState.FAILED
                    return {
                        'success': False,
                        'state': 'failed',
                        'reason': check_result.get('reason', ''),
                        'attempts': attempts
                    }

                # Prepare for replan
                self.state = ExecutionState.REPLANNING
                replan_context = {
                    'check_failure': check_result.get('reason', '')
                }

            except Exception as e:
                attempts.append({
                    'attempt': attempt + 1,
                    'state': 'error',
                    'error': str(e)
                })

        # Replan exhausted
        self.state = ExecutionState.FAILED
        return {
            'success': False,
            'state': 'replan_exhausted',
            'reason': 'Maximum replan attempts reached',
            'attempts': attempts
        }

    def _evaluate_doable(
        self,
        instruction: str,
        scene_data: Optional[Dict],
        context: Dict
    ) -> Dict:
        """Call perception service (stub)"""
        # Would call ROS service
        return {'doable': True, 'confidence': 0.9}

    def _generate_plan(self, instruction: str, context: Dict) -> Dict:
        """Call planning service (stub)"""
        # Would call ROS service
        return {'instruction': instruction, 'actions': []}

    def _execute_plan(self, plan: Dict) -> Dict:
        """Call execution action (stub)"""
        # Would call ROS action
        return {'success': True, 'message': 'Executed'}

    def _check_completion(self, plan: Dict, exec_result: Dict) -> Dict:
        """Call verification service (stub)"""
        # Would call ROS service
        return {'success': True, 'need_replan': False}
