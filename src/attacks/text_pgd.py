"""
Text PGD Attack
文本投影梯度下降攻击
"""
import torch
import torch.nn as nn
from .base_text import BaseTextAttack
from typing import Tuple, Optional


class TextPGD(BaseTextAttack):
    """文本PGD攻击"""

    def __init__(self, model: nn.Module, epsilon: float = 0.1, 
                 alpha: float = 0.01, num_iter: int = 10, device: str = 'cuda'):
        """
        Args:
            model: 目标模型
            epsilon: 最大扰动强度
            alpha: 每步扰动步长
            num_iter: 迭代次数
            device: 计算设备
        """
        super().__init__(model, device)
        self.epsilon = epsilon
        self.alpha = alpha
        self.num_iter = num_iter

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
        # 初始化对抗样本
        adv_embeddings = text_embeddings.clone().detach().to(self.device)
        adv_embeddings.requires_grad = True
        
        labels = labels.to(self.device)

        # PGD迭代
        for i in range(self.num_iter):
            # 前向传播
            outputs = self.model(adv_embeddings)
            
            # 计算损失
            loss = self._compute_loss(outputs, labels, targeted)
            
            # 反向传播
            self.model.zero_grad()
            loss.backward()
            
            # 获取梯度
            grad = adv_embeddings.grad.data
            
            # 更新对抗样本
            if targeted:
                # 目标攻击：减去梯度方向
                adv_embeddings = adv_embeddings - self.alpha * grad.sign()
            else:
                # 无目标攻击：加上梯度方向
                adv_embeddings = adv_embeddings + self.alpha * grad.sign()
            
            # 投影到epsilon球内
            eta = torch.clamp(adv_embeddings - text_embeddings, -self.epsilon, self.epsilon)
            adv_embeddings = (text_embeddings + eta).detach()
            adv_embeddings.requires_grad = True

        # 最终前向传播获取输出
        with torch.no_grad():
            final_outputs = self.model(adv_embeddings)
        
        # 计算扰动指标
        perturbation = adv_embeddings - text_embeddings
        perturbation_l2 = perturbation.norm(p=2, dim=-1).mean().item()
        perturbation_linf = perturbation.abs().max().item()
        
        info = {
            'perturbation_l2': perturbation_l2,
            'perturbation_linf': perturbation_linf,
            'epsilon': self.epsilon,
            'alpha': self.alpha,
            'num_iter': self.num_iter,
            'targeted': targeted
        }
        
        return adv_embeddings.detach(), info