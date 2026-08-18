"""
AuraVLA Perception Module

Implements vision-based scene understanding and task evaluation.
"""

from .vlm_client import VLMClient, NvidiaVLMClient
from .doable_evaluator import DoableEvaluator

__all__ = [
    'VLMClient',
    'NvidiaVLMClient',
    'DoableEvaluator',
]
