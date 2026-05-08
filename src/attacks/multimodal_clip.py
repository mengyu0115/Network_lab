"""
CLIP multimodal attack (image/text/joint) for image-text alignment tasks.
"""
from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F
from torchvision.transforms import ToPILImage


class CLIPMultimodalAttack:
    """
    Gradient-based multimodal attack on CLIP alignment.

    This attacker supports:
    - image-only attack
    - text-feature-only attack
    - joint image + text-feature attack
    """

    def __init__(
        self,
        clip_model,
        processor,
        device: str = "cuda",
        image_epsilon: float = 8.0 / 255.0,
        image_alpha: float = 1.0 / 255.0,
        text_epsilon: float = 0.20,
        text_alpha: float = 0.02,
        num_steps: int = 20,
    ):
        self.clip_model = clip_model
        self.processor = processor
        self.device = device if torch.cuda.is_available() else "cpu"
        self.image_epsilon = image_epsilon
        self.image_alpha = image_alpha
        self.text_epsilon = text_epsilon
        self.text_alpha = text_alpha
        self.num_steps = num_steps

        self.clip_model.to(self.device)
        self.clip_model.eval()
        vision_cfg = getattr(getattr(self.clip_model, "config", None), "vision_config", None)
        self.image_size = int(getattr(vision_cfg, "image_size", 224))
        self.to_pil = ToPILImage()
        image_processor = getattr(self.processor, "image_processor", None)
        image_mean = getattr(image_processor, "image_mean", [0.48145466, 0.4578275, 0.40821073])
        image_std = getattr(image_processor, "image_std", [0.26862954, 0.26130258, 0.27577711])
        self.image_mean = torch.tensor(image_mean, dtype=torch.float32, device=self.device).view(1, 3, 1, 1)
        self.image_std = torch.tensor(image_std, dtype=torch.float32, device=self.device).view(1, 3, 1, 1)

    def _normalize(self, feat: torch.Tensor) -> torch.Tensor:
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
        """Auto-resize image tensor to CLIP required square size."""
        if image.dim() != 4:
            raise ValueError(f"Expected image tensor [B,C,H,W], got shape={tuple(image.shape)}")
        h, w = int(image.shape[-2]), int(image.shape[-1])
        if h == self.image_size and w == self.image_size:
            return image
        return F.interpolate(image, size=(self.image_size, self.image_size), mode="bilinear", align_corners=False)

    def _raw_batch_to_pil(self, image: torch.Tensor):
        pil_images = []
        for item in image.detach().cpu():
            current = item
            if current.shape[0] == 1:
                current = current.repeat(3, 1, 1)
            pil_images.append(self.to_pil(current.clamp(0.0, 1.0)))
        return pil_images

    def _prepare_pixel_values(self, image: torch.Tensor) -> torch.Tensor:
        image = self._match_image_size(image)
        pil_images = self._raw_batch_to_pil(image)
        inputs = self.processor(images=pil_images, return_tensors="pt")
        return inputs["pixel_values"].to(self.device)

    def _denormalize(self, pixel_values: torch.Tensor) -> torch.Tensor:
        return (pixel_values * self.image_std + self.image_mean).clamp(0.0, 1.0)

    def generate(
        self,
        image: torch.Tensor,
        source_text: str,
        target_text: Optional[str] = None,
        mode: str = "joint",
        targeted: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        """
        Args:
            image: Tensor [1,3,H,W], expected in [0,1]
            source_text: original paired text
            target_text: optional target text for targeted attack
            mode: 'image' | 'text' | 'joint'
            targeted: whether to perform targeted attack
        """
        if mode not in {"image", "text", "joint"}:
            raise ValueError("mode must be one of: image, text, joint")
        if targeted and not target_text:
            raise ValueError("target_text is required when targeted=True")

        image = image.to(self.device).detach()
        pixel_values = self._prepare_pixel_values(image)
        source_feat = self._text_feature(source_text).detach()
        target_feat = self._text_feature(target_text).detach() if target_text else None

        adv_image = pixel_values.clone().detach()
        text_delta = torch.zeros_like(source_feat, device=self.device)
        eps_tensor = self.image_epsilon / self.image_std
        alpha_tensor = self.image_alpha / self.image_std
        min_bound = (0.0 - self.image_mean) / self.image_std
        max_bound = (1.0 - self.image_mean) / self.image_std

        source_sim_history = []
        target_sim_history = []

        for _ in range(self.num_steps):
            need_img_grad = mode in {"image", "joint"}
            need_text_grad = mode in {"text", "joint"}

            if need_img_grad:
                adv_image = adv_image.detach().requires_grad_(True)
            else:
                adv_image = adv_image.detach()

            if need_text_grad:
                text_delta = text_delta.detach().requires_grad_(True)
                current_text_feat = self._normalize(source_feat + text_delta)
            else:
                text_delta = text_delta.detach()
                current_text_feat = source_feat

            img_feat = self._normalize(self._as_feature_tensor(self.clip_model.get_image_features(adv_image)))
            source_sim = (img_feat * current_text_feat).sum(dim=-1).mean()

            if targeted:
                target_sim = (img_feat * target_feat).sum(dim=-1).mean()
                # Minimize this loss == maximize target_sim and minimize source_sim.
                loss = -(target_sim - source_sim)
            else:
                target_sim = None
                # Minimize this loss == reduce source alignment.
                loss = source_sim

            grad_vars = []
            if need_img_grad:
                grad_vars.append(adv_image)
            if need_text_grad:
                grad_vars.append(text_delta)

            grads = torch.autograd.grad(loss, grad_vars, allow_unused=True)
            grad_idx = 0

            if need_img_grad:
                grad_img = grads[grad_idx]
                grad_idx += 1
                if grad_img is not None:
                    adv_image = adv_image - alpha_tensor * grad_img.sign()
                    img_pert = torch.clamp(adv_image - pixel_values, -eps_tensor, eps_tensor)
                    adv_image = torch.max(torch.min(pixel_values + img_pert, max_bound), min_bound).detach()

            if need_text_grad:
                grad_txt = grads[grad_idx]
                if grad_txt is not None:
                    text_delta = text_delta - self.text_alpha * grad_txt.sign()
                    text_delta = torch.clamp(text_delta, -self.text_epsilon, self.text_epsilon).detach()

            with torch.no_grad():
                img_feat_eval = self._normalize(self._as_feature_tensor(self.clip_model.get_image_features(adv_image)))
                txt_feat_eval = self._normalize(source_feat + text_delta) if mode in {"text", "joint"} else source_feat
                s_sim = (img_feat_eval * txt_feat_eval).sum(dim=-1).mean().item()
                source_sim_history.append(s_sim)
                if targeted:
                    t_sim = (img_feat_eval * target_feat).sum(dim=-1).mean().item()
                    target_sim_history.append(t_sim)

        adv_text_feat = self._normalize(source_feat + text_delta) if mode in {"text", "joint"} else source_feat
        adv_raw_image = self._denormalize(adv_image)
        base_raw_image = self._denormalize(pixel_values)
        perturbation = (adv_raw_image - base_raw_image).abs()

        info = {
            "mode": mode,
            "targeted": targeted,
            "image_epsilon": self.image_epsilon,
            "text_epsilon": self.text_epsilon,
            "num_steps": self.num_steps,
            "perturbation_l2": perturbation.norm(p=2, dim=(1, 2, 3)).mean().item(),
            "perturbation_linf": perturbation.max().item(),
            "source_similarity_history": source_sim_history,
            "target_similarity_history": target_sim_history,
        }
        return adv_raw_image.detach(), adv_text_feat.detach(), info
