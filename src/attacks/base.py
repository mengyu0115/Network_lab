"""
Base Attack Class
所有攻击算法的基类
"""
import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Tuple, Optional


class BaseAttack(ABC):
    """攻击算法基类"""

    def __init__(self, model: nn.Module, device: str = 'cuda'):
        """
        Args:
            model: 目标模型
            device: 计算设备
        """
        self.model = model
        self.device = device if torch.cuda.is_available() else 'cpu'
        self.model.to(self.device)
        self.model.eval()

    @abstractmethod
    def generate(self, images: torch.Tensor, labels: torch.Tensor,
                 **kwargs) -> Tuple[torch.Tensor, dict]:
        """
        生成对抗样本

        Args:
            images: 原始图像 [B, C, H, W]
            labels: 真实标签 [B]
            **kwargs: 其他参数

        Returns:
            adv_images: 对抗样本
            info: 攻击信息字典
        """
        pass

    def _clip_perturbation(self, images: torch.Tensor,
                          adv_images: torch.Tensor,
                          epsilon: float) -> torch.Tensor:
        """限制扰动范围"""
        perturbation = torch.clamp(adv_images - images, -epsilon, epsilon)
        return torch.clamp(images + perturbation, 0, 1)

    def _compute_loss(self, outputs: torch.Tensor,
                     labels: torch.Tensor,
                     targeted: bool = False) -> torch.Tensor:
        """计算损失"""
        criterion = nn.CrossEntropyLoss()
        loss = criterion(outputs, labels)
        return -loss if targeted else loss
