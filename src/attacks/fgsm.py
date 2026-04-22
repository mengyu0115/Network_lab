"""
FGSM (Fast Gradient Sign Method) Attack
快速梯度符号攻击
"""
import torch
import torch.nn as nn
from typing import Tuple
from .base import BaseAttack


class FGSM(BaseAttack):
    """
    FGSM攻击算法
    论文: Explaining and Harnessing Adversarial Examples (Goodfellow et al., 2015)
    """

    def __init__(self, model: nn.Module, epsilon: float = 0.03, device: str = 'cuda'):
        """
        Args:
            model: 目标模型
            epsilon: 扰动强度
            device: 计算设备
        """
        super().__init__(model, device)
        self.epsilon = epsilon

    def generate(self, images: torch.Tensor, labels: torch.Tensor,
                 targeted: bool = False, target_labels: torch.Tensor = None) -> Tuple[torch.Tensor, dict]:
        """
        生成FGSM对抗样本

        Args:
            images: 原始图像 [B, C, H, W]
            labels: 真实标签 [B]
            targeted: 是否为目标攻击
            target_labels: 目标标签（目标攻击时使用）

        Returns:
            adv_images: 对抗样本
            info: 攻击信息
        """
        images = images.to(self.device)
        labels = labels.to(self.device)

        if targeted and target_labels is not None:
            labels = target_labels.to(self.device)

        # 需要计算梯度
        images.requires_grad = True

        # 前向传播
        outputs = self.model(images)

        # 计算损失
        loss = self._compute_loss(outputs, labels, targeted)

        # 反向传播
        self.model.zero_grad()
        loss.backward()

        # 获取梯度符号
        grad_sign = images.grad.sign()

        # 生成对抗样本
        if targeted:
            adv_images = images - self.epsilon * grad_sign
        else:
            adv_images = images + self.epsilon * grad_sign

        # 裁剪到有效范围
        adv_images = torch.clamp(adv_images, 0, 1).detach()

        # 计算扰动
        perturbation = (adv_images - images.detach()).abs()

        info = {
            'epsilon': self.epsilon,
            'perturbation_l2': perturbation.norm(p=2, dim=(1,2,3)).mean().item(),
            'perturbation_linf': perturbation.max().item(),
            'targeted': targeted
        }

        return adv_images, info
