"""
Text Attack Class
文本攻击算法基类
"""
import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Tuple, Optional, List, Union


class BaseTextAttack(ABC):
    """文本攻击算法基类"""

    def __init__(self, model: nn.Module, device: str = 'cuda'):
        """
        Args:
            model: 目标模型（接受文本输入或文本嵌入）
            device: 计算设备
        """
        self.model = model
        self.device = device if torch.cuda.is_available() else 'cpu'
        self.model.to(self.device)
        self.model.eval()

    @abstractmethod
    def generate(self, text_embeddings: torch.Tensor, 
                 labels: torch.Tensor,
                 **kwargs) -> Tuple[torch.Tensor, dict]:
        """
        生成对抗文本嵌入

        Args:
            text_embeddings: 原始文本嵌入 [B, seq_len, embed_dim] 或 [B, embed_dim]
            labels: 真实标签 [B]
            **kwargs: 其他参数

        Returns:
            adv_embeddings: 对抗文本嵌入
            info: 攻击信息字典
        """
        pass

    def _clip_perturbation(self, embeddings: torch.Tensor,
                          adv_embeddings: torch.Tensor,
                          epsilon: float) -> torch.Tensor:
        """限制扰动范围"""
        perturbation = torch.clamp(adv_embeddings - embeddings, -epsilon, epsilon)
        return embeddings + perturbation

    def _compute_loss(self, outputs: torch.Tensor,
                     labels: torch.Tensor,
                     targeted: bool = False) -> torch.Tensor:
        """计算损失"""
        criterion = nn.CrossEntropyLoss()
        loss = criterion(outputs, labels)
        return -loss if targeted else loss