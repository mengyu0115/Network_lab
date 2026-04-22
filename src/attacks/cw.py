"""
C&W (Carlini & Wagner) Attack
C&W攻击算法
"""
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Tuple
from .base import BaseAttack


class CarliniWagner(BaseAttack):
    """
    C&W L2攻击算法
    论文: Towards Evaluating the Robustness of Neural Networks (Carlini & Wagner, 2017)
    """

    def __init__(self, model: nn.Module, c: float = 1.0,
                 kappa: float = 0, learning_rate: float = 0.01,
                 num_iter: int = 100, device: str = 'cuda'):
        """
        Args:
            model: 目标模型
            c: 损失权重系数
            kappa: 置信度参数
            learning_rate: 学习率
            num_iter: 迭代次数
            device: 计算设备
        """
        super().__init__(model, device)
        self.c = c
        self.kappa = kappa
        self.learning_rate = learning_rate
        self.num_iter = num_iter

    def generate(self, images: torch.Tensor, labels: torch.Tensor,
                 targeted: bool = False, target_labels: torch.Tensor = None) -> Tuple[torch.Tensor, dict]:
        """
        生成C&W对抗样本

        Args:
            images: 原始图像 [B, C, H, W]
            labels: 真实标签 [B]
            targeted: 是否为目标攻击
            target_labels: 目标标签

        Returns:
            adv_images: 对抗样本
            info: 攻击信息
        """
        images = images.to(self.device)
        labels = labels.to(self.device)

        if targeted and target_labels is not None:
            labels = target_labels.to(self.device)

        batch_size = images.shape[0]
        num_classes = self.model(images).shape[1]

        # 使用tanh空间优化
        w = self._inverse_tanh_space(images)
        w = w.clone().detach().requires_grad_(True)

        optimizer = optim.Adam([w], lr=self.learning_rate)

        loss_history = []
        best_adv = images.clone()
        best_l2 = [float('inf')] * batch_size

        for i in range(self.num_iter):
            # 转换回图像空间
            adv_images = self._tanh_space(w)

            # 前向传播
            outputs = self.model(adv_images)

            # 计算L2距离
            l2_dist = (adv_images - images).pow(2).sum(dim=(1,2,3))

            # 计算分类损失
            f_loss = self._f_loss(outputs, labels, targeted)

            # 总损失
            loss = l2_dist.sum() + self.c * f_loss.sum()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_history.append(loss.item())

            # 更新最佳对抗样本
            for j in range(batch_size):
                if l2_dist[j] < best_l2[j] and self._is_successful(outputs[j], labels[j], targeted):
                    best_l2[j] = l2_dist[j].item()
                    best_adv[j] = adv_images[j].detach()

        perturbation = (best_adv - images).abs()

        info = {
            'c': self.c,
            'kappa': self.kappa,
            'num_iter': self.num_iter,
            'perturbation_l2': perturbation.norm(p=2, dim=(1,2,3)).mean().item(),
            'perturbation_linf': perturbation.max().item(),
            'loss_history': loss_history,
            'targeted': targeted
        }

        return best_adv, info

    def _inverse_tanh_space(self, x: torch.Tensor) -> torch.Tensor:
        """将[0,1]映射到tanh空间"""
        return torch.atanh((x * 2 - 1) * 0.999999)

    def _tanh_space(self, w: torch.Tensor) -> torch.Tensor:
        """将tanh空间映射回[0,1]"""
        return (torch.tanh(w) + 1) / 2

    def _f_loss(self, outputs: torch.Tensor, labels: torch.Tensor, targeted: bool) -> torch.Tensor:
        """计算C&W损失函数"""
        batch_size = outputs.shape[0]
        real = outputs.gather(1, labels.unsqueeze(1)).squeeze(1)

        # 获取除真实标签外的最大logit
        other = outputs.clone()
        other.scatter_(1, labels.unsqueeze(1), float('-inf'))
        other_max = other.max(1)[0]

        if targeted:
            # 目标攻击: 最大化目标类别概率
            loss = torch.clamp(other_max - real + self.kappa, min=0)
        else:
            # 无目标攻击: 最小化真实类别概率
            loss = torch.clamp(real - other_max + self.kappa, min=0)

        return loss

    def _is_successful(self, output: torch.Tensor, label: torch.Tensor, targeted: bool) -> bool:
        """判断攻击是否成功"""
        pred = output.argmax()
        if targeted:
            return pred == label
        else:
            return pred != label
