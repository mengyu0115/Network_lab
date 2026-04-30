"""
Evaluation Metrics Module
评估指标模块
"""
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple
from sklearn.metrics import accuracy_score, confusion_matrix
import torch.nn.functional as F


class AttackEvaluator:
    """攻击效果评估器"""

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

        self.model_type = getattr(model, 'model_type', 'vision')

    def evaluate(self, original_images: torch.Tensor,
                adversarial_images: torch.Tensor,
                true_labels: torch.Tensor,
                targeted: bool = False,
                target_labels: torch.Tensor = None) -> Dict:
        """
        全面评估攻击效果

        Args:
            original_images: 原始图像
            adversarial_images: 对抗样本
            true_labels: 真实标签
            targeted: 是否为目标攻击
            target_labels: 目标标签

        Returns:
            metrics: 评估指标字典
        """
        original_images = original_images.to(self.device)
        adversarial_images = adversarial_images.to(self.device)
        true_labels = true_labels.to(self.device)

        with torch.no_grad():
            # 原始预测
            orig_outputs = self.model(original_images)
            orig_preds = orig_outputs.argmax(dim=1)

            # 对抗样本预测
            adv_outputs = self.model(adversarial_images)
            adv_preds = adv_outputs.argmax(dim=1)

        # 计算各项指标
        metrics = {}

        # 1. 攻击成功率
        metrics['attack_success_rate'] = self._compute_attack_success_rate(
            orig_preds, adv_preds, true_labels, targeted, target_labels
        )
        metrics['prediction_flip_rate'] = metrics['attack_success_rate']

        # 2. 扰动强度
        perturbation_metrics = self._compute_perturbation_metrics(
            original_images, adversarial_images
        )
        metrics.update(perturbation_metrics)

        # 3. 感知质量
        perceptual_metrics = self._compute_perceptual_metrics(
            original_images, adversarial_images
        )
        metrics.update(perceptual_metrics)

        # 4. 置信度变化
        confidence_metrics = self._compute_confidence_metrics(
            orig_outputs, adv_outputs, true_labels
        )
        metrics.update(confidence_metrics)

        # 5. 预测结果
        metrics['original_predictions'] = orig_preds.cpu().detach().numpy()
        metrics['adversarial_predictions'] = adv_preds.cpu().detach().numpy()
        metrics['true_labels'] = true_labels.cpu().detach().numpy()

        return metrics

    def _compute_attack_success_rate(self, orig_preds: torch.Tensor,
                                     adv_preds: torch.Tensor,
                                     true_labels: torch.Tensor,
                                     targeted: bool,
                                     target_labels: torch.Tensor = None) -> float:
        """计算攻击成功率"""
        if targeted and target_labels is not None:
            # 目标攻击：对抗样本被分类为目标类别
            success = (adv_preds == target_labels).float().mean().item()
        else:
            # 无目标攻击：对抗样本分类错误
            success = (adv_preds != orig_preds).float().mean().item()

        return success * 100  # 转换为百分比

    def _compute_perturbation_metrics(self, original: torch.Tensor,
                                     adversarial: torch.Tensor) -> Dict:
        """计算扰动强度指标"""
        perturbation = adversarial - original

        metrics = {
            'perturbation_l0': (perturbation != 0).float().mean().item(),
            'perturbation_l2': perturbation.norm(p=2, dim=(1,2,3)).mean().item(),
            'perturbation_linf': perturbation.abs().max().item(),
            'perturbation_mean': perturbation.abs().mean().item(),
        }

        return metrics

    def _compute_perceptual_metrics(self, original: torch.Tensor,
                                   adversarial: torch.Tensor) -> Dict:
        """计算感知质量指标"""
        # SSIM (简化版)
        ssim = self._compute_ssim(original, adversarial)

        # PSNR
        mse = F.mse_loss(adversarial, original)
        psnr = 10 * torch.log10(1.0 / (mse + 1e-10))

        metrics = {
            'ssim': ssim.item(),
            'psnr': psnr.item(),
        }

        return metrics

    def _compute_ssim(self, img1: torch.Tensor, img2: torch.Tensor,
                     window_size: int = 11) -> torch.Tensor:
        """计算SSIM (简化版)"""
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2

        mu1 = F.avg_pool2d(img1, window_size, stride=1, padding=window_size//2)
        mu2 = F.avg_pool2d(img2, window_size, stride=1, padding=window_size//2)

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2

        sigma1_sq = F.avg_pool2d(img1 * img1, window_size, stride=1, padding=window_size//2) - mu1_sq
        sigma2_sq = F.avg_pool2d(img2 * img2, window_size, stride=1, padding=window_size//2) - mu2_sq
        sigma12 = F.avg_pool2d(img1 * img2, window_size, stride=1, padding=window_size//2) - mu1_mu2

        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
                   ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

        return ssim_map.mean()

    def _compute_confidence_metrics(self, orig_outputs: torch.Tensor,
                                   adv_outputs: torch.Tensor,
                                   true_labels: torch.Tensor) -> Dict:
        """计算置信度变化指标"""
        orig_probs = F.softmax(orig_outputs, dim=1)
        adv_probs = F.softmax(adv_outputs, dim=1)

        # 真实类别的置信度
        orig_confidence = orig_probs.gather(1, true_labels.unsqueeze(1)).squeeze()
        adv_confidence = adv_probs.gather(1, true_labels.unsqueeze(1)).squeeze()

        metrics = {
            'original_confidence': orig_confidence.mean().item(),
            'adversarial_confidence': adv_confidence.mean().item(),
            'confidence_drop': (orig_confidence - adv_confidence).mean().item(),
        }

        return metrics

    def evaluate_transferability(self, adversarial_images: torch.Tensor,
                                true_labels: torch.Tensor,
                                target_models: List[nn.Module]) -> Dict:
        """
        评估对抗样本的迁移性

        Args:
            adversarial_images: 对抗样本
            true_labels: 真实标签
            target_models: 目标模型列表

        Returns:
            transfer_metrics: 迁移性指标
        """
        adversarial_images = adversarial_images.to(self.device)
        true_labels = true_labels.to(self.device)

        transfer_rates = []

        for model in target_models:
            model.to(self.device)
            model.eval()

            with torch.no_grad():
                outputs = model(adversarial_images)
                preds = outputs.argmax(dim=1)

            # 计算攻击成功率
            success_rate = (preds != true_labels).float().mean().item()
            transfer_rates.append(success_rate * 100)

        metrics = {
            'transfer_rates': transfer_rates,
            'average_transfer_rate': np.mean(transfer_rates),
        }

        return metrics


def compute_confusion_matrix(true_labels: np.ndarray,
                            predictions: np.ndarray) -> np.ndarray:
    """计算混淆矩阵"""
    return confusion_matrix(true_labels, predictions)


def compute_accuracy(true_labels: np.ndarray,
                    predictions: np.ndarray) -> float:
    """计算准确率"""
    return accuracy_score(true_labels, predictions) * 100


class CLIPEvaluator:
    """CLIP模型专用评估器"""

    def __init__(self, clip_model, device: str = 'cpu'):
        import torch.nn.functional as F
        self.model = clip_model
        self.device = device if torch.cuda.is_available() else 'cpu'
        self.F = F

    def compute_feature_similarity(self, original_images: torch.Tensor,
                                adversarial_images: torch.Tensor) -> Dict:
        """计算CLIP特征余弦相似度"""
        original_images = original_images.to(self.device)
        adversarial_images = adversarial_images.to(self.device)

        with torch.no_grad():
            orig_features = self.model.clip_model.get_image_features(original_images)
            adv_features = self.model.clip_model.get_image_features(adversarial_images)

            orig_features = orig_features / orig_features.norm(dim=-1, keepdim=True)
            adv_features = adv_features / adv_features.norm(dim=-1, keepdim=True)

            cosine_sim = (orig_features * adv_features).sum(dim=-1)

        metrics = {
            'clip_cosine_similarity': cosine_sim.mean().item(),
            'clip_feature_l2': (orig_features - adv_features).norm(dim=-1).mean().item(),
        }

        return metrics

    def evaluate(self, original_images: torch.Tensor,
                  adversarial_images: torch.Tensor,
                  text_prompts: List[str] = None) -> Dict:
        """完整评估"""
        metrics = self.compute_feature_similarity(original_images, adversarial_images)

        if text_prompts:
            if getattr(self.model, 'processor', None) is None:
                raise ValueError("CLIP processor is missing on the wrapped model.")

            original_images = original_images.to(self.device)
            adversarial_images = adversarial_images.to(self.device)

            with torch.no_grad():
                orig_inputs = self.model.processor(
                    text=text_prompts, images=original_images, return_tensors="pt", padding=True
                ).to(self.device)

                adv_inputs = self.model.processor(
                    text=text_prompts, images=adversarial_images, return_tensors="pt", padding=True
                ).to(self.device)

                orig_logits = self.model.clip_model(**orig_inputs).logits_per_image
                adv_logits = self.model.clip_model(**adv_inputs).logits_per_image

            metrics['text_image_similarity_orig'] = orig_logits.mean().item()
            metrics['text_image_similarity_adv'] = adv_logits.mean().item()
            metrics['text_similarity_change'] = (orig_logits - adv_logits).abs().mean().item()

        return metrics
