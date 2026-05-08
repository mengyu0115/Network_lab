"""
PGD (Projected Gradient Descent) Attack
投影梯度下降攻击
"""
import torch
import torch.nn as nn
from typing import Tuple
from .base import BaseAttack


class PGD(BaseAttack):
    """
    PGD攻击算法
    论文: Towards Deep Learning Models Resistant to Adversarial Attacks (Madry et al., 2018)
    """

    def __init__(self, model: nn.Module, epsilon: float = 0.03,
                 alpha: float = 0.01, num_iter: int = 10, device: str = 'cuda'):
        """
        Args:
            model: 目标模型
            epsilon: 最大扰动强度
            alpha: 每步扰动大小
            num_iter: 迭代次数
            device: 计算设备
        """
        super().__init__(model, device)
        self.epsilon = epsilon
        self.alpha = alpha
        self.num_iter = num_iter

    def generate(self, images: torch.Tensor, labels: torch.Tensor,
                 targeted: bool = False, target_labels: torch.Tensor = None,
                 random_start: bool = True) -> Tuple[torch.Tensor, dict]:
        """
        生成PGD对抗样本

        Args:
            images: 原始图像 [B, C, H, W]
            labels: 真实标签 [B]
            targeted: 是否为目标攻击
            target_labels: 目标标签
            random_start: 是否随机初始化

        Returns:
            adv_images: 对抗样本
            info: 攻击信息
        """
        images = images.to(self.device)
        labels = labels.to(self.device)

        if targeted and target_labels is not None:
            labels = target_labels.to(self.device)

        # 初始化对抗样本
        adv_images = images.clone().detach()

        if random_start:
            # 随机初始化
            adv_images = adv_images + torch.empty_like(adv_images).uniform_(-self.epsilon, self.epsilon)
            adv_images = torch.clamp(adv_images, 0, 1)

        loss_history = []

        # 迭代攻击
        for i in range(self.num_iter):
            adv_images = adv_images.detach().requires_grad_(True)

            # 前向传播
            outputs = self.model(adv_images)

            # 计算损失
            loss = self._compute_loss(outputs, labels, targeted)
            loss_history.append(loss.item())

            # 反向传播
            self.model.zero_grad()
            loss.backward()

            # 更新对抗样本
            grad = adv_images.grad.sign()

            if targeted:
                adv_images = adv_images.detach() - self.alpha * grad
            else:
                adv_images = adv_images.detach() + self.alpha * grad

            # 投影到epsilon球内
            adv_images = self._clip_perturbation(images, adv_images, self.epsilon).detach()

        # 计算最终扰动
        perturbation = (adv_images - images).abs()

        info = {
            'epsilon': self.epsilon,
            'alpha': self.alpha,
            'num_iter': self.num_iter,
            'perturbation_l2': perturbation.norm(p=2, dim=(1,2,3)).mean().item(),
            'perturbation_linf': perturbation.max().item(),
            'loss_history': loss_history,
            'targeted': targeted
        }

        return adv_images, info
