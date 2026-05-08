"""
BLIP image attack for multimodal target prioritization.
"""
from __future__ import annotations

import re
from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F
from torchvision.transforms import ToPILImage


class BLIPCaptionAttack:
    """Iterative pixel-space attack on BLIP image captioning."""

    def __init__(
        self,
        caption_model,
        processor,
        device: str = "cuda",
        epsilon: float = 8.0 / 255.0,
        alpha: float = 2.0 / 255.0,
        num_steps: int = 10,
        num_restarts: int = 3,
        source_weight: float = 0.35,
        **_: object,
    ):
        self.caption_model = caption_model
        self.base_model = getattr(caption_model, "base_model", caption_model)
        self.processor = processor or getattr(caption_model, "processor", None)
        self.device = device if torch.cuda.is_available() else "cpu"
        self.epsilon = epsilon
        self.alpha = alpha
        self.num_steps = num_steps
        self.num_restarts = max(int(num_restarts), 1)
        self.source_weight = float(source_weight)
        self.to_pil = ToPILImage()

        image_processor = getattr(self.processor, "image_processor", None)
        image_mean = getattr(image_processor, "image_mean", [0.5, 0.5, 0.5])
        image_std = getattr(image_processor, "image_std", [0.5, 0.5, 0.5])
        self.image_mean = torch.tensor(image_mean, dtype=torch.float32, device=self.device).view(1, 3, 1, 1)
        self.image_std = torch.tensor(image_std, dtype=torch.float32, device=self.device).view(1, 3, 1, 1)

        self.caption_model.to(self.device)
        self.caption_model.eval()

    def _raw_batch_to_pil(self, image: torch.Tensor):
        pil_images = []
        for item in image.detach().cpu():
            current = item
            if current.shape[0] == 1:
                current = current.repeat(3, 1, 1)
            pil_images.append(self.to_pil(current.clamp(0.0, 1.0)))
        return pil_images

    def _prepare_pixel_values(self, image: torch.Tensor) -> torch.Tensor:
        pil_images = self._raw_batch_to_pil(image)
        inputs = self.processor(images=pil_images, return_tensors="pt")
        return inputs["pixel_values"].to(self.device)

    def _encode_text(self, text: str) -> Dict[str, torch.Tensor]:
        text_inputs = self.processor(text=[text], return_tensors="pt", padding=True, truncation=True)
        return {k: v.to(self.device) for k, v in text_inputs.items()}

    def _denormalize(self, pixel_values: torch.Tensor) -> torch.Tensor:
        return (pixel_values * self.image_std + self.image_mean).clamp(0.0, 1.0)

    @staticmethod
    def _tokenize_content(text: str) -> set[str]:
        tokens = re.findall(r"[a-z0-9]+", (text or "").lower())
        stopwords = {
            "a", "an", "the", "of", "in", "on", "at", "to", "for", "with", "and", "or",
            "photo", "picture", "image", "scene", "this", "that", "is", "are", "be",
        }
        return {token for token in tokens if token not in stopwords and len(token) > 1}

    def _caption_score(self, caption: str, source_text: str, target_text: str | None, targeted: bool) -> float:
        caption_tokens = self._tokenize_content(caption)
        source_tokens = self._tokenize_content(source_text)
        target_tokens = self._tokenize_content(target_text or "")
        source_overlap = len(caption_tokens & source_tokens)
        target_overlap = len(caption_tokens & target_tokens)
        if targeted:
            return float(target_overlap - source_overlap)
        return float(-source_overlap)

    def _compute_caption_loss(
        self,
        pixel_values: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        outputs = self.base_model(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=input_ids,
        )
        return outputs.loss

    def _generate_caption(self, pixel_values: torch.Tensor, max_new_tokens: int = 30) -> str:
        with torch.no_grad():
            generated = self.base_model.generate(pixel_values=pixel_values, max_new_tokens=max_new_tokens)
        decoded = self.processor.batch_decode(generated, skip_special_tokens=True)
        return decoded[0].strip() if decoded else ""

    def generate(
        self,
        image: torch.Tensor,
        source_text: str = "",
        target_text: Optional[str] = None,
        targeted: bool = True,
    ) -> Tuple[torch.Tensor, Dict]:
        if targeted and not target_text:
            raise ValueError("target_text is required when targeted=True")

        image = image.to(self.device).detach()
        pixel_values = self._prepare_pixel_values(image)
        adv_image = pixel_values.clone().detach()
        source_caption = source_text.strip() or self._generate_caption(pixel_values)

        if targeted:
            caption_text = target_text
        else:
            caption_text = source_caption

        target_inputs = self._encode_text(caption_text)
        target_input_ids = target_inputs["input_ids"]
        target_attention_mask = target_inputs.get("attention_mask")
        source_inputs = self._encode_text(source_caption)
        source_input_ids = source_inputs["input_ids"]
        source_attention_mask = source_inputs.get("attention_mask")

        eps_tensor = self.epsilon / self.image_std
        alpha_tensor = self.alpha / self.image_std
        min_bound = (0.0 - self.image_mean) / self.image_std
        max_bound = (1.0 - self.image_mean) / self.image_std

        best_adv = adv_image.clone().detach()
        best_caption = self._generate_caption(pixel_values)
        best_score = self._caption_score(best_caption, source_caption, target_text, targeted)
        best_loss_history = []

        for restart_idx in range(self.num_restarts):
            if restart_idx == 0:
                current_adv = adv_image.clone().detach()
            else:
                noise = torch.empty_like(pixel_values).uniform_(-1.0, 1.0) * eps_tensor
                current_adv = torch.max(torch.min(pixel_values + noise, max_bound), min_bound).detach()

            loss_history = []
            for _ in range(self.num_steps):
                current_adv = current_adv.detach().requires_grad_(True)
                target_loss = self._compute_caption_loss(
                    current_adv,
                    target_input_ids,
                    target_attention_mask,
                )

                if targeted:
                    source_loss = self._compute_caption_loss(
                        current_adv,
                        source_input_ids,
                        source_attention_mask,
                    )
                    loss = target_loss - self.source_weight * source_loss
                else:
                    loss = -self._compute_caption_loss(
                        current_adv,
                        source_input_ids,
                        source_attention_mask,
                    )

                loss_history.append(float(loss.item()))
                grad = torch.autograd.grad(loss, current_adv, retain_graph=False, create_graph=False)[0]
                current_adv = current_adv - alpha_tensor * grad.sign()
                delta = torch.clamp(current_adv - pixel_values, -eps_tensor, eps_tensor)
                current_adv = torch.max(torch.min(pixel_values + delta, max_bound), min_bound).detach()

            current_caption = self._generate_caption(current_adv)
            current_score = self._caption_score(current_caption, source_caption, target_text, targeted)
            if current_score > best_score:
                best_score = current_score
                best_adv = current_adv.detach()
                best_caption = current_caption
                best_loss_history = loss_history

        adv_raw = self._denormalize(best_adv)
        base_raw = self._denormalize(pixel_values)
        perturbation = (adv_raw - base_raw).abs()

        info = {
            "targeted": targeted,
            "epsilon": self.epsilon,
            "alpha": self.alpha,
            "num_steps": self.num_steps,
            "num_restarts": self.num_restarts,
            "source_weight": self.source_weight,
            "loss_history": best_loss_history,
            "perturbation_l2": perturbation.norm(p=2, dim=(1, 2, 3)).mean().item(),
            "perturbation_linf": perturbation.max().item(),
            "original_caption": source_caption,
            "adversarial_caption": best_caption,
            "caption_score": best_score,
        }
        return adv_raw.detach(), info
