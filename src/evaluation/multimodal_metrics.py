"""
Metrics for CLIP multimodal attacks.
"""
from typing import Dict, Optional

import torch
import torch.nn.functional as F


class CLIPMultimodalEvaluator:
    """Evaluate image-text alignment degradation/targeting under attack."""

    def __init__(self, clip_model, processor, device: str = "cuda"):
        self.clip_model = clip_model
        self.processor = processor
        self.device = device if torch.cuda.is_available() else "cpu"
        self.clip_model.to(self.device)
        self.clip_model.eval()
        vision_cfg = getattr(getattr(self.clip_model, "config", None), "vision_config", None)
        self.image_size = int(getattr(vision_cfg, "image_size", 224))

    @staticmethod
    def _normalize(feat: torch.Tensor) -> torch.Tensor:
        return F.normalize(feat, p=2, dim=-1)

    @staticmethod
    def _as_feature_tensor(feat):
        if isinstance(feat, torch.Tensor):
            return feat
        if hasattr(feat, "pooler_output") and feat.pooler_output is not None:
            return feat.pooler_output
        if hasattr(feat, "last_hidden_state"):
            return feat.last_hidden_state.mean(dim=1)
        raise TypeError("Unsupported CLIP feature output type.")

    def _text_feature(self, text: str) -> torch.Tensor:
        text_inputs = self.processor(text=[text], return_tensors="pt", padding=True).to(self.device)
        with torch.no_grad():
            feat = self._as_feature_tensor(self.clip_model.get_text_features(**text_inputs))
        return self._normalize(feat)

    def _match_image_size(self, image: torch.Tensor) -> torch.Tensor:
        if image.dim() != 4:
            raise ValueError(f"Expected image tensor [B,C,H,W], got shape={tuple(image.shape)}")
        h, w = int(image.shape[-2]), int(image.shape[-1])
        if h == self.image_size and w == self.image_size:
            return image
        return F.interpolate(image, size=(self.image_size, self.image_size), mode="bilinear", align_corners=False)

    def evaluate(
        self,
        original_image: torch.Tensor,
        adversarial_image: torch.Tensor,
        source_text: str,
        target_text: Optional[str] = None,
        adversarial_text_feature: Optional[torch.Tensor] = None,
        targeted: bool = False,
    ) -> Dict:
        original_image = self._match_image_size(original_image.to(self.device))
        adversarial_image = self._match_image_size(adversarial_image.to(self.device))

        with torch.no_grad():
            orig_img_feat = self._normalize(self._as_feature_tensor(self.clip_model.get_image_features(original_image)))
            adv_img_feat = self._normalize(self._as_feature_tensor(self.clip_model.get_image_features(adversarial_image)))

        source_text_feat = self._text_feature(source_text)
        adv_text_feat = self._normalize(adversarial_text_feature.to(self.device)) if adversarial_text_feature is not None else source_text_feat

        orig_source_sim = (orig_img_feat * source_text_feat).sum(dim=-1).mean().item()
        adv_source_sim = (adv_img_feat * adv_text_feat).sum(dim=-1).mean().item()

        metrics = {
            "orig_source_similarity": orig_source_sim,
            "adv_source_similarity": adv_source_sim,
            "similarity_drop": orig_source_sim - adv_source_sim,
            "attack_success": adv_source_sim < orig_source_sim,
        }

        if targeted and target_text:
            target_feat = self._text_feature(target_text)
            orig_target_sim = (orig_img_feat * target_feat).sum(dim=-1).mean().item()
            adv_target_sim = (adv_img_feat * target_feat).sum(dim=-1).mean().item()
            targeted_success = adv_target_sim > adv_source_sim
            metrics.update(
                {
                    "orig_target_similarity": orig_target_sim,
                    "adv_target_similarity": adv_target_sim,
                    "target_similarity_gain": adv_target_sim - orig_target_sim,
                    "targeted_attack_success": targeted_success,
                }
            )

        perturbation = (adversarial_image - original_image).abs()
        metrics.update(
            {
                "perturbation_l2": perturbation.norm(p=2, dim=(1, 2, 3)).mean().item(),
                "perturbation_linf": perturbation.max().item(),
                "perturbation_mean": perturbation.mean().item(),
            }
        )
        return metrics
