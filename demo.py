"""
示例脚本: 运行对抗攻击实验
"""
import torch
import sys
import os
import numpy as np # 新增 numpy 导入，用于数据类型判断
import multiprocessing as mp

# Windows 上设置启动方法
if sys.platform == 'win32':
    mp.set_start_method('spawn', force=True)

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.attacks import FGSM, PGD, CarliniWagner, TextFGSM, TextPGD
from src.models import load_model, ModelLoader
from src.data_manager import DatasetManager
from src.evaluation import AttackEvaluator

# 延迟导入可视化，避免 Windows multiprocessing 冲突
def run_experiment_with_visualization():
    from src.visualization import Visualizer
    return Visualizer()
import matplotlib.pyplot as plt

# 解决中文显示问题
plt.rcParams['font.sans-serif'] = ['SimHei']  # Windows 系统指定默认字体为黑体
# 如果你是 Mac 系统，请将上面那行换成： plt.rcParams['font.sans-serif'] = ['PingFang SC']

# 解决保存图像是负号 '-' 显示为方块的问题
plt.rcParams['axes.unicode_minus'] = False

def run_attack_experiment():
    """运行完整的攻击实验"""

    print("=" * 60)
    print("多模态模型攻击实验")
    print("=" * 60)

    # 1. 配置参数
    model_name = 'resnet50'
    dataset_name = 'cifar10'
    attack_method = 'PGD'
    batch_size = 32
    num_samples = 100
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print(f"\n配置:")
    print(f"  模型: {model_name}")
    print(f"  数据集: {dataset_name}")
    print(f"  攻击方法: {attack_method}")
    print(f"  设备: {device}")

    # 2. 加载模型
    print(f"\n正在加载模型 {model_name}...")
    model = load_model(model_name, pretrained=True, device=device)
    print("✓ 模型加载成功")

    # 3. 加载数据
    print(f"\n正在加载数据集 {dataset_name}...")
    data_manager = DatasetManager()
    dataloader = data_manager.load_dataset(
        dataset_name,
        split='test',
        batch_size=batch_size,
        shuffle=False
    )
    print("✓ 数据集加载成功")

    # 4. 创建攻击器
    print(f"\n创建 {attack_method} 攻击器...")
    if attack_method == 'FGSM':
        attacker = FGSM(model, epsilon=0.03, device=device)
    elif attack_method == 'PGD':
        attacker = PGD(model, epsilon=0.03, alpha=0.01, num_iter=10, device=device)
    else:
        attacker = CarliniWagner(model, c=1.0, num_iter=100, device=device)
    print("✓ 攻击器创建成功")

    # 5. 执行攻击
    print(f"\n开始生成对抗样本...")
    all_original = []
    all_adversarial = []
    all_labels = []
    all_predictions = []

    sample_count = 0
    for images, labels in dataloader:
        if sample_count >= num_samples:
            break

        images = images.to(device)
        labels = labels.to(device)

        # 生成对抗样本
        adv_images, info = attacker.generate(images, labels)

        # 预测
        with torch.no_grad():
            orig_preds = model(images).argmax(dim=1)
            adv_preds = model(adv_images).argmax(dim=1)

        all_original.append(images.cpu())
        all_adversarial.append(adv_images.cpu())
        all_labels.append(labels.cpu())
        all_predictions.append(adv_preds.cpu())

        sample_count += len(images)
        print(f"  已处理: {sample_count}/{num_samples} 样本")

    # 合并结果
    all_original = torch.cat(all_original)
    all_adversarial = torch.cat(all_adversarial)
    all_labels = torch.cat(all_labels)
    all_predictions = torch.cat(all_predictions)

    print("✓ 对抗样本生成完成")

    # 6. 评估攻击效果
    print(f"\n评估攻击效果...")
    evaluator = AttackEvaluator(model, device=device)
    metrics = evaluator.evaluate(
        all_original,
        all_adversarial,
        all_labels
    )

    print("\n" + "=" * 60)
    print("评估结果:")
    print("=" * 60)
    print(f"攻击成功率: {metrics['attack_success_rate']:.2f}%")
    print(f"L2 扰动: {metrics['perturbation_l2']:.4f}")
    print(f"L∞ 扰动: {metrics['perturbation_linf']:.4f}")
    print(f"SSIM: {metrics['ssim']:.4f}")
    print(f"PSNR: {metrics['psnr']:.2f} dB")
    print(f"原始置信度: {metrics['original_confidence']:.4f}")
    print(f"对抗置信度: {metrics['adversarial_confidence']:.4f}")
    print(f"置信度下降: {metrics['confidence_drop']:.4f}")

    print("\n" + "-" * 60)
    print("攻击样例对比（前10个）:")
    print("-" * 60)
    # 【已修改】：为了兼容 ImageNet 的 1000 分类输出，取消中文字符串匹配，直接打印预测的数字索引
    for i in range(min(10, len(metrics['original_predictions']))):
        true_label = int(metrics['true_labels'][i])
        orig_pred = int(metrics['original_predictions'][i])
        adv_pred = int(metrics['adversarial_predictions'][i])
        success = "✓ 成功" if orig_pred != adv_pred else "✗ 失败"
        print(f"样例 {i+1:<4} 真实标签:{true_label:<5} 原预测:{orig_pred:<5} 对抗预测:{adv_pred:<5} {success:<10}")
    print("=" * 60)

    # ---------------------------------------------------------
    # 【修复逻辑】：清洗 metrics 数据格式，防止 JSON 序列化报错
    # ---------------------------------------------------------
    sanitized_metrics = {}
    for key, value in metrics.items():
        if isinstance(value, np.ndarray):
            sanitized_metrics[key] = value.tolist()
        elif isinstance(value, torch.Tensor):
            sanitized_metrics[key] = value.cpu().detach().numpy().tolist()
        elif isinstance(value, (np.floating, float)):
            sanitized_metrics[key] = float(value)
        elif isinstance(value, (np.integer, int)):
            sanitized_metrics[key] = int(value)
        elif isinstance(value, list):
            # 将列表中可能存在的 tensor/numpy 标量转为普通标量
            sanitized_metrics[key] = [
                float(i) if isinstance(i, (np.floating, float)) else
                int(i) if isinstance(i, (np.integer, int)) else
                i.item() if isinstance(i, torch.Tensor) else i 
                for i in value
            ]
        else:
            sanitized_metrics[key] = value
    # ---------------------------------------------------------

    # 7. 保存结果
    print(f"\n保存实验结果...")
    save_dir = data_manager.save_adversarial_samples(
        all_original[:10],
        all_labels[:10],
        all_adversarial[:10],
        all_predictions[:10],
        attack_method,
        metadata={
            'model': model_name,
            'dataset': dataset_name,
            'num_samples': num_samples,
            **sanitized_metrics  # 使用清洗后的 metrics
        }
    )
    print(f"✓ 结果已保存到: {save_dir}")

    # 8. 可视化
    print(f"\n生成可视化图表...")
    visualizer = run_experiment_with_visualization()

    # 绘制对抗样本对比
    for i in range(min(3, len(all_original))):
        visualizer.plot_adversarial_comparison(
            all_original[i].detach(),
            all_adversarial[i].detach(),
            (all_adversarial[i] - all_original[i]).abs().detach(),
            f"Class {metrics['original_predictions'][i]}",
            f"Class {metrics['adversarial_predictions'][i]}",
            f"Class {all_labels[i].item()}",
            save_path=os.path.join(save_dir, f'comparison_{i}.png')
        )

    print("✓ 可视化完成")
    print("\n实验完成!")


if __name__ == '__main__':
    run_attack_experiment()