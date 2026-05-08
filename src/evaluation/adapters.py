"""
Unified multimodal evaluation adapters.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import torch
from torchvision.transforms import ToPILImage

from src.models import ModelLoader
from .metrics import CLIPEvaluator


@dataclass(frozen=True)
class AdapterSpec:
    key: str
    display_name: str
    task: str
    enabled: bool
    local_only: bool = True


class BaseEvaluationAdapter(ABC):
    key = "base"
    display_name = "Base Adapter"
    task = "generic"
    enabled = True

    def __init__(self, device: str = "cpu"):
        self.device = device if torch.cuda.is_available() else "cpu"

    @abstractmethod
    def evaluate(
        self,
        original_images: torch.Tensor,
        adversarial_images: torch.Tensor,
        **kwargs,
    ) -> Dict[str, Any]:
        """Evaluate original/adversarial image pairs."""


class CLIPEvaluationAdapter(BaseEvaluationAdapter):
    key = "clip"
    display_name = "CLIP 图文对齐评测"
    task = "similarity"

    def __init__(self, model_name: str, device: str = "cpu"):
        super().__init__(device=device)
        self.model_name = model_name
        self.model = ModelLoader.load_model(
            model_name,
            pretrained=True,
            device=self.device,
        )
        self.evaluator = CLIPEvaluator(self.model, device=self.device)

    def evaluate(
        self,
        original_images: torch.Tensor,
        adversarial_images: torch.Tensor,
        **kwargs,
    ) -> Dict[str, Any]:
        text_prompts = kwargs.get("text_prompts")
        metrics = self.evaluator.evaluate(
            original_images=original_images,
            adversarial_images=adversarial_images,
            text_prompts=text_prompts,
        )
        metrics["adapter"] = self.key
        metrics["model_name"] = self.model_name
        return metrics


class BLIPCaptionEvaluationAdapter(BaseEvaluationAdapter):
    key = "blip"
    display_name = "BLIP 图像描述评测"
    task = "captioning"

    def __init__(self, model_name: str = "blip-base", device: str = "cpu"):
        super().__init__(device=device)
        self.model_name = model_name
        self.model = ModelLoader.load_caption_model(model_name, device=self.device)
        self._to_pil = ToPILImage()

    def _tensor_batch_to_pil(self, images: torch.Tensor) -> List[Any]:
        images = images.detach().cpu()
        pil_images: List[Any] = []
        for image in images:
            current = image
            if current.dim() != 3:
                raise ValueError("Expected image tensor with shape [C, H, W].")
            if current.shape[0] == 1:
                current = current.repeat(3, 1, 1)
            pil_images.append(self._to_pil(current.clamp(0.0, 1.0)))
        return pil_images

    def _generate_captions(
        self,
        images: torch.Tensor,
        prompt: Optional[str] = None,
        max_new_tokens: int = 30,
    ) -> List[str]:
        pil_images = self._tensor_batch_to_pil(images)
        if prompt:
            prompt_batch = [prompt] * len(pil_images)
            inputs = self.model.processor(
                images=pil_images,
                text=prompt_batch,
                return_tensors="pt",
                padding=True,
            ).to(self.device)
        else:
            inputs = self.model.processor(
                images=pil_images,
                return_tensors="pt",
            ).to(self.device)

        with torch.no_grad():
            generated = self.model.base_model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
            )

        captions = self.model.processor.batch_decode(generated, skip_special_tokens=True)
        return [caption.strip() for caption in captions]

    @staticmethod
    def _normalize_caption(text: str) -> str:
        return " ".join(text.lower().strip().split())

    def evaluate(
        self,
        original_images: torch.Tensor,
        adversarial_images: torch.Tensor,
        **kwargs,
    ) -> Dict[str, Any]:
        prompt = kwargs.get("prompt")
        max_new_tokens = int(kwargs.get("max_new_tokens", 30))

        original_captions = self._generate_captions(
            original_images,
            prompt=prompt,
            max_new_tokens=max_new_tokens,
        )
        adversarial_captions = self._generate_captions(
            adversarial_images,
            prompt=prompt,
            max_new_tokens=max_new_tokens,
        )

        changed = [
            self._normalize_caption(orig) != self._normalize_caption(adv)
            for orig, adv in zip(original_captions, adversarial_captions)
        ]
        length_delta = [
            abs(len(orig) - len(adv))
            for orig, adv in zip(original_captions, adversarial_captions)
        ]

        return {
            "adapter": self.key,
            "model_name": self.model_name,
            "prompt": prompt or "",
            "original_captions": original_captions,
            "adversarial_captions": adversarial_captions,
            "caption_change_rate": float(sum(changed) / max(len(changed), 1) * 100.0),
            "average_caption_length_delta": float(sum(length_delta) / max(len(length_delta), 1)),
        }


class ReservedCloudEvaluationAdapter(BaseEvaluationAdapter):
    enabled = False

    def __init__(self, key: str, display_name: str, device: str = "cpu"):
        super().__init__(device=device)
        self.key = key
        self.display_name = display_name

    def evaluate(
        self,
        original_images: torch.Tensor,
        adversarial_images: torch.Tensor,
        **kwargs,
    ) -> Dict[str, Any]:
        raise NotImplementedError(
            f"{self.display_name} adapter is reserved for future authorized integration."
        )


class EvaluationAdapterRegistry:
    """Registry for local evaluators and future reserved providers."""

    SPECS = {
        "clip": AdapterSpec(
            key="clip",
            display_name="CLIP 图文对齐评测",
            task="similarity",
            enabled=True,
        ),
        "blip": AdapterSpec(
            key="blip",
            display_name="BLIP 图像描述评测",
            task="captioning",
            enabled=True,
        ),
        "qwen_vl": AdapterSpec(
            key="qwen_vl",
            display_name="Qwen-VL 评测适配器（预留）",
            task="cloud_captioning",
            enabled=False,
            local_only=False,
        ),
        "glm4v": AdapterSpec(
            key="glm4v",
            display_name="GLM-4V 评测适配器（预留）",
            task="cloud_captioning",
            enabled=False,
            local_only=False,
        ),
    }

    @classmethod
    def list_specs(cls, enabled_only: bool = False) -> List[AdapterSpec]:
        specs = list(cls.SPECS.values())
        if enabled_only:
            specs = [spec for spec in specs if spec.enabled]
        return specs

    @classmethod
    def create(
        cls,
        adapter_key: str,
        device: str = "cpu",
        model_name: Optional[str] = None,
    ) -> BaseEvaluationAdapter:
        if adapter_key == "clip":
            if not model_name:
                raise ValueError("CLIP adapter requires model_name.")
            return CLIPEvaluationAdapter(model_name=model_name, device=device)
        if adapter_key == "blip":
            return BLIPCaptionEvaluationAdapter(model_name=model_name or "blip-base", device=device)
        if adapter_key == "qwen_vl":
            return ReservedCloudEvaluationAdapter("qwen_vl", "Qwen-VL", device=device)
        if adapter_key == "glm4v":
            return ReservedCloudEvaluationAdapter("glm4v", "GLM-4V", device=device)
        raise ValueError(f"Unknown adapter key: {adapter_key}")


def load_evaluation_adapter(
    adapter_key: str,
    device: str = "cpu",
    model_name: Optional[str] = None,
) -> BaseEvaluationAdapter:
    """Factory entrypoint used by the web layer."""
    return EvaluationAdapterRegistry.create(
        adapter_key=adapter_key,
        device=device,
        model_name=model_name,
    )
