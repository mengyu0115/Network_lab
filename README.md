# 图像对抗攻击与安全评估平台

## 项目概述

本项目当前以图片模态为主线，面向图像分类模型开展对抗攻击与安全评估。系统实现了多种图像对抗样本生成算法，支持对主流视觉模型进行攻击测试，并提供评估指标、实验记录和可视化展示。

## 核心功能

1. **攻击场景设计** - 支持图像分类任务的白盒对抗攻击场景
2. **数据集管理** - 支持 CIFAR-10、MNIST、自定义图片、实验记录和攻击前后对比
3. **攻击算法实现** - FGSM、PGD、C&W等梯度优化对抗攻击算法
4. **效果评估** - 攻击成功率、扰动强度、语义一致性等多维度评估
5. **可视化展示** - Web界面支持攻击配置、实验运行和结果分析

## 技术栈

- **后端**: Python 3.8+, PyTorch, TorchVision
- **前端**: Streamlit / Flask
- **模型**: ResNet-18、ResNet-50 等视觉模型
- **可视化**: Matplotlib, Plotly

## 项目结构

```
multimodal-attack-platform/
├── data/                   # 数据集目录
│   ├── raw/               # 原始数据
│   ├── processed/         # 处理后数据
│   └── adversarial/       # 对抗样本
├── src/                   # 源代码
│   ├── attacks/           # 攻击算法实现
│   ├── models/            # 模型加载与管理
│   ├── evaluation/        # 评估指标
│   ├── data_manager/      # 数据管理
│   └── visualization/     # 可视化模块
├── web/                   # Web界面
├── experiments/           # 实验记录
├── docs/                  # 文档
│   ├── 算法说明.md
│   └── 总结报告.md
├── requirements.txt       # 依赖包
└── README.md
```

## 快速开始

> 📖 详细安装和运行指南请查看 [INSTALL.md](INSTALL.md)

### 安装依赖

```bash
pip install -r requirements.txt
```

### 运行平台

```bash
streamlit run web/app.py
```

### 训练标准数据集模型

使用 CIFAR-10 或 MNIST 做批量攻击前，需要先训练对应的 10 类分类模型，否则平台会停止标准数据集实验，避免生成不可信结果。

```bash
python scripts/train_classifier.py --dataset cifar10 --model resnet18 --epochs 5
python scripts/train_classifier.py --dataset mnist --model resnet18 --epochs 3
```

训练完成后，权重会保存到 `data/checkpoints/`，Web 页面会自动加载。

### 运行攻击实验

```python
from src.attacks import FGSM
from src.models import load_model
from src.evaluation import evaluate_attack

model = load_model('resnet50')
attacker = FGSM(model, epsilon=0.03)
results = attacker.generate(images, labels)
metrics = evaluate_attack(results)
```

## 实验流程

1. 选择目标模型和数据集
2. 配置攻击参数（扰动强度、迭代次数等）
3. 生成对抗样本
4. 评估攻击效果
5. 查看可视化结果和统计报告

## 评估指标

- **攻击成功率 (ASR)**: 成功误导模型的样本比例
- **平均扰动强度 (L∞/L2)**: 对抗扰动的范数度量
- **感知质量 (SSIM/PSNR)**: 对抗样本与原始样本的相似度
- **查询次数**: 黑盒攻击所需的模型查询次数
- **迁移成功率**: 对抗样本在其他模型上的有效性

## 成果文档

- [算法说明文档](docs/算法说明.md) - 详细的攻击算法原理和实现
- [总结报告](docs/总结报告.md) - 实验结果分析和研究总结

## 许可证

本项目仅用于学术研究和安全评估，请勿用于恶意攻击。
