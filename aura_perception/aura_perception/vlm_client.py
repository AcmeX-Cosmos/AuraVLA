"""
VLM Client for AuraVLA Perception

Provides interface to Vision-Language Models for scene understanding.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Dict, Any
import json
import os
from urllib.request import Request, urlopen
from urllib.error import HTTPError
import base64
from io import BytesIO


@dataclass
class VLMConfig:
    """VLM Configuration"""
    model: str = "nvidia/nemotron-nano-12b-v2-vl"
    base_url: str = "https://integrate.api.nvidia.com/v1"
    api_key: str = ""
    max_tokens: int = 768
    temperature: float = 0.2
    top_p: float = 0.9
    request_timeout_sec: float = 300.0
    max_retries: int = 1
    image_max_edge: int = 448


class VLMClient(ABC):
    """Abstract VLM Client"""

    @abstractmethod
    def infer(self, prompt: str, image=None) -> Dict[str, Any]:
        """Perform inference with VLM"""
        pass


class NvidiaVLMClient(VLMClient):
    """NVIDIA VLM Client Implementation"""

    def __init__(self, config: VLMConfig):
        self.config = config
        if not self.config.api_key:
            self.config.api_key = os.environ.get('NVIDIA_API_KEY', '')

    def infer(self, prompt: str, image=None) -> Dict[str, Any]:
        """
        Perform VLM inference

        Args:
            prompt: Text prompt
            image: Optional image (path, numpy array, or PIL Image)

        Returns:
            Dict with inference results
        """
        messages = [
            {
                "role": "user",
                "content": prompt
            }
        ]

        # Add image if provided
        if image is not None:
            image_b64 = self._encode_image(image)
            messages[0]["content"] = [
                {"type": "text", "text": prompt},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}
                }
            ]

        # Prepare request
        payload = {
            "model": self.config.model,
            "messages": messages,
            "max_tokens": self.config.max_tokens,
            "temperature": self.config.temperature,
            "top_p": self.config.top_p,
            "stream": False
        }

        # Make API call
        for attempt in range(self.config.max_retries):
            try:
                response = self._call_api(payload)
                content = response['choices'][0]['message']['content']

                # Try to parse as JSON
                try:
                    return json.loads(content)
                except json.JSONDecodeError:
                    # Extract JSON from markdown code block
                    if '```json' in content:
                        json_str = content.split('```json')[1].split('```')[0].strip()
                        return json.loads(json_str)
                    return {"raw_response": content}

            except Exception as e:
                if attempt < self.config.max_retries - 1:
                    continue
                raise RuntimeError(f"VLM inference failed: {e}")

    def _call_api(self, payload: Dict) -> Dict:
        """Call NVIDIA API"""
        headers = {
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json"
        }

        url = f"{self.config.base_url}/chat/completions"
        request = Request(
            url,
            data=json.dumps(payload).encode('utf-8'),
            headers=headers,
            method='POST'
        )

        with urlopen(request, timeout=self.config.request_timeout_sec) as response:
            return json.loads(response.read().decode('utf-8'))

    def _encode_image(self, image) -> str:
        """Encode image to base64"""
        try:
            from PIL import Image
            import numpy as np

            # Handle different input types
            if isinstance(image, str):
                img = Image.open(image)
            elif isinstance(image, np.ndarray):
                img = Image.fromarray(image)
            elif isinstance(image, Image.Image):
                img = image
            else:
                raise ValueError(f"Unsupported image type: {type(image)}")

            # Resize if needed
            if max(img.size) > self.config.image_max_edge:
                ratio = self.config.image_max_edge / max(img.size)
                new_size = tuple(int(dim * ratio) for dim in img.size)
                img = img.resize(new_size, Image.Resampling.LANCZOS)

            # Convert to base64
            buffer = BytesIO()
            img.convert('RGB').save(buffer, format='JPEG', quality=85, optimize=True)
            return base64.b64encode(buffer.getvalue()).decode('utf-8')

        except ImportError:
            raise RuntimeError("PIL required for image encoding. Install: pip install Pillow")
