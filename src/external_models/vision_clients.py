"""
Black-box vision model clients for authorized robustness evaluation.

The project uses these clients to compare the same benign prompt on an
original image and an adversarial image. API keys are read from environment
variables and are never stored in experiment metadata.
"""
from __future__ import annotations

import base64
import hashlib
import json
import os
from dataclasses import dataclass
from io import BytesIO
from typing import Dict, List, Optional
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from PIL import Image


class VisionModelError(RuntimeError):
    """Raised when a remote vision model call cannot be completed."""


@dataclass(frozen=True)
class VisionModelResult:
    provider: str
    model: str
    prompt: str
    output: str


@dataclass(frozen=True)
class ProviderSpec:
    key: str
    display_name: str
    env_var: str
    default_model: str


PROVIDER_SPECS: Dict[str, ProviderSpec] = {
    "dashscope": ProviderSpec(
        key="dashscope",
        display_name="Alibaba Qwen-VL",
        env_var="DASHSCOPE_API_KEY",
        default_model="qwen-vl-plus-latest",
    ),
    "openai": ProviderSpec(
        key="openai",
        display_name="OpenAI GPT-4o",
        env_var="OPENAI_API_KEY",
        default_model="gpt-4o",
    ),
    "gemini": ProviderSpec(
        key="gemini",
        display_name="Google Gemini 2.5 Pro",
        env_var="GEMINI_API_KEY",
        default_model="gemini-2.5-pro",
    ),
    "anthropic": ProviderSpec(
        key="anthropic",
        display_name="Anthropic Claude",
        env_var="ANTHROPIC_API_KEY",
        default_model="claude-sonnet-4-20250514",
    ),
}


def list_vision_providers() -> List[ProviderSpec]:
    return list(PROVIDER_SPECS.values())


def _image_to_base64(image: Image.Image, image_format: str = "PNG") -> str:
    buffer = BytesIO()
    image.convert("RGB").save(buffer, format=image_format)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def _image_bytes(image: Image.Image, image_format: str = "PNG") -> bytes:
    buffer = BytesIO()
    image.convert("RGB").save(buffer, format=image_format)
    return buffer.getvalue()


def _cache_dir() -> str:
    root = os.environ.get("BLACKBOX_CACHE_DIR", os.path.join("data", "blackbox_cache"))
    os.makedirs(root, exist_ok=True)
    return root


def _cache_key(provider: str, model: str, prompt: str, image: Image.Image) -> str:
    digest = hashlib.sha256()
    digest.update(provider.encode("utf-8"))
    digest.update(model.encode("utf-8"))
    digest.update(prompt.encode("utf-8"))
    digest.update(_image_bytes(image))
    return digest.hexdigest()


def _read_cached_result(provider: str, model: str, prompt: str, image: Image.Image) -> Optional[VisionModelResult]:
    path = os.path.join(_cache_dir(), f"{_cache_key(provider, model, prompt, image)}.json")
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    return VisionModelResult(
        provider=payload["provider"],
        model=payload["model"],
        prompt=payload["prompt"],
        output=payload["output"],
    )


def _write_cached_result(result: VisionModelResult, image: Image.Image) -> None:
    path = os.path.join(_cache_dir(), f"{_cache_key(result.provider, result.model, result.prompt, image)}.json")
    payload = {
        "provider": result.provider,
        "model": result.model,
        "prompt": result.prompt,
        "output": result.output,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def _post_json(url: str, payload: Dict, headers: Dict[str, str], timeout: int = 90) -> Dict:
    body = json.dumps(payload).encode("utf-8")
    request = Request(url, data=body, headers=headers, method="POST")
    try:
        with urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise VisionModelError(f"HTTP {exc.code}: {detail}") from exc
    except URLError as exc:
        raise VisionModelError(f"Network error: {exc.reason}") from exc
    except TimeoutError as exc:
        raise VisionModelError("Request timed out.") from exc


class BaseVisionClient:
    provider_key = "base"

    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None):
        spec = PROVIDER_SPECS[self.provider_key]
        self.api_key = api_key or os.environ.get(spec.env_var, "")
        self.model = model or spec.default_model
        if not self.api_key:
            raise VisionModelError(f"Missing API key. Set {spec.env_var}.")

    def ask_image(self, image: Image.Image, prompt: str, use_cache: bool = True) -> VisionModelResult:
        if use_cache:
            cached = _read_cached_result(self.provider_key, self.model, prompt, image)
            if cached is not None:
                return cached
        result = self._ask_image_uncached(image, prompt)
        if use_cache:
            _write_cached_result(result, image)
        return result

    def _ask_image_uncached(self, image: Image.Image, prompt: str) -> VisionModelResult:
        raise NotImplementedError


class OpenAIVisionClient(BaseVisionClient):
    provider_key = "openai"

    def _ask_image_uncached(self, image: Image.Image, prompt: str) -> VisionModelResult:
        image_b64 = _image_to_base64(image)
        payload = {
            "model": self.model,
            "input": [
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": prompt},
                        {
                            "type": "input_image",
                            "image_url": f"data:image/png;base64,{image_b64}",
                        },
                    ],
                }
            ],
        }
        data = _post_json(
            "https://api.openai.com/v1/responses",
            payload,
            {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
        )
        output = data.get("output_text", "")
        if not output:
            parts = []
            for item in data.get("output", []):
                for content in item.get("content", []):
                    text = content.get("text")
                    if text:
                        parts.append(text)
            output = "\n".join(parts)
        return VisionModelResult("openai", self.model, prompt, output.strip())


class DashScopeVisionClient(BaseVisionClient):
    provider_key = "dashscope"

    def _ask_image_uncached(self, image: Image.Image, prompt: str) -> VisionModelResult:
        image_b64 = _image_to_base64(image)
        base_url = os.environ.get(
            "DASHSCOPE_BASE_URL",
            "https://dashscope.aliyuncs.com/compatible-mode/v1",
        ).rstrip("/")
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{image_b64}",
                            },
                        },
                        {"type": "text", "text": prompt},
                    ],
                }
            ],
            "max_tokens": 512,
        }
        data = _post_json(
            f"{base_url}/chat/completions",
            payload,
            {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
        )
        output = ""
        choices = data.get("choices", [])
        if choices:
            output = choices[0].get("message", {}).get("content", "")
        return VisionModelResult("dashscope", self.model, prompt, str(output).strip())


class GeminiVisionClient(BaseVisionClient):
    provider_key = "gemini"

    def _ask_image_uncached(self, image: Image.Image, prompt: str) -> VisionModelResult:
        image_b64 = _image_to_base64(image)
        payload = {
            "contents": [
                {
                    "parts": [
                        {"text": prompt},
                        {
                            "inline_data": {
                                "mime_type": "image/png",
                                "data": image_b64,
                            }
                        },
                    ]
                }
            ]
        }
        data = _post_json(
            f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent?key={self.api_key}",
            payload,
            {"Content-Type": "application/json"},
        )
        parts = []
        for candidate in data.get("candidates", []):
            for part in candidate.get("content", {}).get("parts", []):
                text = part.get("text")
                if text:
                    parts.append(text)
        return VisionModelResult("gemini", self.model, prompt, "\n".join(parts).strip())


class AnthropicVisionClient(BaseVisionClient):
    provider_key = "anthropic"

    def _ask_image_uncached(self, image: Image.Image, prompt: str) -> VisionModelResult:
        image_b64 = _image_to_base64(image)
        payload = {
            "model": self.model,
            "max_tokens": 512,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/png",
                                "data": image_b64,
                            },
                        },
                        {"type": "text", "text": prompt},
                    ],
                }
            ],
        }
        data = _post_json(
            "https://api.anthropic.com/v1/messages",
            payload,
            {
                "x-api-key": self.api_key,
                "anthropic-version": "2023-06-01",
                "Content-Type": "application/json",
            },
        )
        output = "\n".join(
            item.get("text", "")
            for item in data.get("content", [])
            if item.get("type") == "text"
        )
        return VisionModelResult("anthropic", self.model, prompt, output.strip())


def create_vision_client(provider: str, model: Optional[str] = None) -> BaseVisionClient:
    provider = provider.strip().lower()
    if provider == "dashscope":
        return DashScopeVisionClient(model=model)
    if provider == "openai":
        return OpenAIVisionClient(model=model)
    if provider == "gemini":
        return GeminiVisionClient(model=model)
    if provider == "anthropic":
        return AnthropicVisionClient(model=model)
    raise VisionModelError(f"Unsupported provider: {provider}")
