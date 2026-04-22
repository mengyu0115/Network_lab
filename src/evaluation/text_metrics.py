"""
Text Attack Evaluation Metrics
文本攻击评估指标
"""
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple
from .metrics import AttackEvaluator


class TextAttackEvaluator:
    """文本攻击效果评估器"""

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

    def evaluate(self, original_embeddings: torch.Tensor,
                adversarial_embeddings: torch.Tensor,
                true_labels: torch.Tensor,
                targeted: bool = False,
                target_labels: torch.Tensor = None) -> Dict:
        """
        评估文本攻击效果

        Args:
            original_embeddings: 原始文本嵌入
            adversarial_embeddings: 对抗文本嵌入
            true_labels: 真实标签
            targeted: 是否为目标攻击
            target_labels: 目标标签

        Returns:
            metrics: 评估指标字典
        """
        original_embeddings = original_embeddings.to(self.device)
        adversarial_embeddings = adversarial_embeddings.to(self.device)
        true_labels = true_labels.to(self.device)

        with torch.no_grad():
            # 原始预测
            orig_outputs = self.model(original_embeddings)
            orig_preds = orig_outputs.argmax(dim=1)

            # 对抗样本预测
            adv_outputs = self.model(adversarial_embeddings)
            adv_preds = adv_outputs.argmax(dim=1)

        # 计算各项指标
        metrics = {}

        # 1. 攻击成功率
        if targeted and target_labels is not None:
            success = (adv_preds == target_labels).float().mean().item()
        else:
            success = (adv_preds != true_labels).float().mean().item()
        metrics['attack_success_rate'] = success * 100

        # 2. 扰动强度
        perturbation = adversarial_embeddings - original_embeddings
        metrics['perturbation_l2'] = perturbation.norm(p=2, dim=-1).mean().item()
        metrics['perturbation_linf'] = perturbation.abs().max().item()
        metrics['perturbation_mean'] = perturbation.abs().mean().item()

        # 3. 预测结果
        metrics['original_predictions'] = orig_preds.cpu().detach().numpy()
        metrics['adversarial_predictions'] = adv_preds.cpu().detach().numpy()
        metrics['true_labels'] = true_labels.cpu().detach().numpy()

        return metrics


# 扩展原有的 AttackEvaluator 以支持文本嵌入
def evaluate_text_attack(model: nn.Module, original_embeddings: torch.Tensor,
                        adversarial_embeddings: torch.Tensor, true_labels: torch.Tensor,
                        **kwargs) -> Dict:
    """便捷函数：评估文本攻击"""
    evaluator = TextAttackEvaluator(model, device=original_embeddings.device.type)
    return evaluator.evaluate(original_embeddings, adversarial_embeddings, true_labels, **kwargs)