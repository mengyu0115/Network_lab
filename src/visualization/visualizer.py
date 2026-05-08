"""
Visualization Module
"""
import matplotlib.pyplot as plt
import numpy as np
import torch
import plotly.graph_objects as go
import plotly.express as px
from typing import List, Dict, Optional
import seaborn as sns


class Visualizer:
    """Visualization Tool Class"""

    def __init__(self, figsize=(12, 8)):
        """
        Args:
            figsize: figure size
        """
        self.figsize = figsize
        plt.style.use('seaborn-v0_8-darkgrid')

    def plot_adversarial_comparison(self, original: torch.Tensor,
                                    adversarial: torch.Tensor,
                                    perturbation: torch.Tensor,
                                    orig_pred: str, adv_pred: str,
                                    true_label: str,
                                    save_path: Optional[str] = None):
        """
        Plot comparison of original, adversarial and perturbation

        Args:
            original: original image
            adversarial: adversarial image
            perturbation: perturbation
            orig_pred: original prediction
            adv_pred: adversarial prediction
            true_label: true label
            save_path: save path
        """
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # Convert to numpy
        orig_img = self._tensor_to_image(original)
        adv_img = self._tensor_to_image(adversarial)
        pert_img = self._tensor_to_image(perturbation)

        # Original image
        axes[0].imshow(orig_img)
        axes[0].set_title(f'Original\nPred: {orig_pred}\nTrue: {true_label}', fontsize=12)
        axes[0].axis('off')

        # 对抗样本
        axes[1].imshow(adv_img)
        axes[1].set_title(f'Adversarial\nPred: {adv_pred}', fontsize=12)
        axes[1].axis('off')

        # 扰动
        axes[2].imshow(pert_img, cmap='hot')
        axes[2].set_title('Perturbation', fontsize=12)
        axes[2].axis('off')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        else:
            plt.show()

        plt.close()

    def plot_attack_success_rate(self, attack_names: List[str],
                                success_rates: List[float],
                                save_path: Optional[str] = None):
        """
        Plot attack success rate comparison

        Args:
            attack_names: list of attack method names
            success_rates: list of success rates
            save_path: save path
        """
        fig, ax = plt.subplots(figsize=self.figsize)

        colors = plt.cm.viridis(np.linspace(0, 1, len(attack_names)))
        bars = ax.bar(attack_names, success_rates, color=colors, alpha=0.8)

        ax.set_ylabel('Attack Success Rate (%)', fontsize=12)
        ax.set_xlabel('Attack Method', fontsize=12)
        ax.set_title('Attack Success Rate Comparison', fontsize=14, fontweight='bold')
        ax.set_ylim(0, 100)

        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.1f}%',
                   ha='center', va='bottom', fontsize=10)

        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        else:
            plt.show()

        plt.close()

    def plot_perturbation_metrics(self, metrics: Dict[str, List[float]],
                                 save_path: Optional[str] = None):
        """
        Plot perturbation metrics comparison

        Args:
            metrics: metrics dict {'attack_name': [l2, linf, ...]}
            save_path: save path
        """
        attack_names = list(metrics.keys())
        metric_types = ['L2', 'L∞', 'Mean']

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        for idx, metric_type in enumerate(metric_types):
            values = [metrics[name][idx] for name in attack_names]

            axes[idx].bar(attack_names, values, alpha=0.7, color='steelblue')
            axes[idx].set_title(f'{metric_type} Perturbation', fontsize=12, fontweight='bold')
            axes[idx].set_ylabel('Value', fontsize=10)
            axes[idx].tick_params(axis='x', rotation=45)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        else:
            plt.show()

        plt.close()

    def plot_confusion_matrix(self, cm: np.ndarray, class_names: List[str],
                            save_path: Optional[str] = None):
        """
        Plot confusion matrix

        Args:
            cm: confusion matrix
            class_names: class names
            save_path: save path
        """
        fig, ax = plt.subplots(figsize=self.figsize)

        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=class_names, yticklabels=class_names,
                   ax=ax, cbar_kws={'label': 'Count'})

        ax.set_xlabel('Predicted Label', fontsize=12)
        ax.set_ylabel('True Label', fontsize=12)
        ax.set_title('Confusion Matrix', fontsize=14, fontweight='bold')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        else:
            plt.show()

        plt.close()

    def plot_loss_history(self, loss_history: List[float],
                         attack_name: str,
                         save_path: Optional[str] = None):
        """
        Plot loss curve

        Args:
            loss_history: loss history
            attack_name: attack name
            save_path: save path
        """
        fig, ax = plt.subplots(figsize=self.figsize)

        ax.plot(loss_history, linewidth=2, color='crimson')
        ax.set_xlabel('Iteration', fontsize=12)
        ax.set_ylabel('Loss', fontsize=12)
        ax.set_title(f'{attack_name} Loss Curve', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        else:
            plt.show()

        plt.close()

    def plot_transferability_heatmap(self, transfer_matrix: np.ndarray,
                                    source_models: List[str],
                                    target_models: List[str],
                                    save_path: Optional[str] = None):
        """
        Plot transferability heatmap

        Args:
            transfer_matrix: transfer matrix [source_models x target_models]
            source_models: source model list
            target_models: target model list
            save_path: save path
        """
        fig, ax = plt.subplots(figsize=self.figsize)

        sns.heatmap(transfer_matrix, annot=True, fmt='.1f',
                   cmap='YlOrRd', xticklabels=target_models,
                   yticklabels=source_models, ax=ax,
                   cbar_kws={'label': 'Transfer Rate (%)'})

        ax.set_xlabel('Target Model', fontsize=12)
        ax.set_ylabel('Source Model', fontsize=12)
        ax.set_title('Transferability Analysis', fontsize=14, fontweight='bold')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        else:
            plt.show()

        plt.close()

    def plot_interactive_metrics(self, metrics_data: Dict,
                                save_path: Optional[str] = None):
        """
        Create interactive metrics chart using Plotly

        Args:
            metrics_data: metrics data
            save_path: save path
        """
        fig = go.Figure()

        for attack_name, metrics in metrics_data.items():
            fig.add_trace(go.Scatter(
                x=list(metrics.keys()),
                y=list(metrics.values()),
                mode='lines+markers',
                name=attack_name,
                line=dict(width=2),
                marker=dict(size=8)
            ))

        fig.update_layout(
            title='Attack Performance Comparison',
            xaxis_title='Metric',
            yaxis_title='Value',
            hovermode='x unified',
            template='plotly_white',
            font=dict(size=12)
        )

        if save_path:
            fig.write_html(save_path)
        else:
            fig.show()

    def _tensor_to_image(self, tensor: torch.Tensor) -> np.ndarray:
        """Convert tensor to displayable image"""
        if tensor.dim() == 4:
            tensor = tensor[0]

        img = tensor.cpu().detach().numpy()

        if img.shape[0] in [1, 3]:
            img = np.transpose(img, (1, 2, 0))

        if img.shape[2] == 1:
            img = img.squeeze(2)

        img = np.clip(img, 0, 1)

        return img
