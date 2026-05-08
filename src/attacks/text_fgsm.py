"""
Text FGSM Attack
文本快速梯度符号攻击
"""
import torch
import torch.nn as nn
from .base_text import BaseTextAttack
from typing import Tuple, Optional


class TextFGSM(BaseTextAttack):
    """文本FGSM攻击"""

    def __init__(self, model: nn.Module, epsilon: float = 0.1, device: str = 'cuda'):
        """
        Args:
            model: 目标模型
            epsilon: 扰动强度
            device: 计算设备
        """
        super().__init__(model, device)
        self.epsilon = epsilon

    def generate(self, text_embeddings: torch.Tensor, 
                 labels: torch.Tensor,
                 targeted: bool = False,
                 **kwargs) -> Tuple[torch.Tensor, dict]:
        """
        生成对抗文本嵌入

        Args:
            text_embeddings: 原始文本嵌入 [B, embed_dim]
            labels: 真实标签 [B]
            targeted: 是否为目标攻击
            **kwargs: 其他参数

        Returns:
            adv_embeddings: 对抗文本嵌入
            info: 攻击信息字典
        """
        # 确保输入是可训练的
        text_embeddings = text_embeddings.clone().detach().to(self.device)
        text_embeddings.requires_grad = True
        
        labels = labels.to(self.device)

        # 前向传播
        outputs = self.model(text_embeddings)
        
        # 计算损失
        loss = self._compute_loss(outputs, labels, targeted)
        
        # 反向传播
        self.model.zero_grad()
        loss.backward()
        
        # 获取梯度
        grad = text_embeddings.grad.data
        
        # 生成对抗样本
        if targeted:
            # 目标攻击：减去梯度方向
            adv_embeddings = text_embeddings - self.epsilon * grad.sign()
        else:
            # 无目标攻击：加上梯度方向
            adv_embeddings = text_embeddings + self.epsilon * grad.sign()
        
        # 限制扰动范围（可选）
        # adv_embeddings = self._clip_perturbation(text_embeddings, adv_embeddings, self.epsilon)
        
        # 计算扰动指标
        perturbation = adv_embeddings - text_embeddings
        perturbation_l2 = perturbation.norm(p=2, dim=-1).mean().item()
        perturbation_linf = perturbation.abs().max().item()
        
        info = {
            'perturbation_l2': perturbation_l2,
            'perturbation_linf': perturbation_linf,
            'epsilon': self.epsilon,
            'targeted': targeted
        }
        
        return adv_embeddings.detach(), info