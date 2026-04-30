# 快速开始指南

## 安装步骤

### 1. 环境准备

确保已安装Python 3.8或以上版本:

```bash
python --version
```

### 2. 安装依赖

```bash
pip install -r requirements.txt
```

如果使用GPU加速,确保已安装CUDA和对应版本的PyTorch。

### 3. 验证安装

```bash
python -c "import torch; print(torch.__version__)"
python -c "import torchvision; print(torchvision.__version__)"
```

## 使用方法

> 当前 Web 入口聚焦图片模态攻击。CIFAR-10/MNIST 批量攻击需要先训练对应微调权重；自定义图片可直接使用 ImageNet 预训练模型演示。

### 方法1: Web界面 (推荐)

启动Streamlit应用:

```bash
streamlit run web/app.py
```

浏览器会自动打开 http://localhost:8501

**操作步骤**:
1. 在侧边栏选择模型 (如 resnet50)
2. 选择数据集（自定义图片可直接演示；CIFAR-10/MNIST 需先训练微调权重）
3. 选择攻击方法 (如 PGD)
4. 调整攻击参数
5. 上传图像或使用标准数据集
6. 点击"开始攻击"按钮
7. 查看结果和评估指标

### 标准数据集模型训练

如果要使用 CIFAR-10 或 MNIST 做可信的批量攻击，先运行：

```bash
python scripts/train_classifier.py --dataset cifar10 --model resnet18 --epochs 5
python scripts/train_classifier.py --dataset mnist --model resnet18 --epochs 3
```

训练脚本会把最优权重保存到 `data/checkpoints/`。保存后重新启动 Web 页面，平台会自动加载对应权重。

### 方法2: 命令行脚本

运行示例脚本:

```bash
python demo.py
```

该脚本会:
- 加载ResNet-50模型
- 使用CIFAR-10数据集
- 执行PGD攻击
- 评估攻击效果
- 保存结果和可视化


### 方法3: Python代码

```python
from src.attacks import FGSM
from src.models import load_model
from src.evaluation import AttackEvaluator
import torch

# 加载模型
model = load_model('resnet50', pretrained=True)

# 准备数据 (示例)
images = torch.rand(10, 3, 224, 224)  # 10张随机图像
labels = torch.randint(0, 1000, (10,))  # 随机标签

# 创建攻击器
attacker = FGSM(model, epsilon=0.03)

# 生成对抗样本
adv_images, info = attacker.generate(images, labels)

# 评估
evaluator = AttackEvaluator(model)
metrics = evaluator.evaluate(images, adv_images, labels)

print(f"攻击成功率: {metrics['attack_success_rate']:.2f}%")
```

## 常见问题

### Q1: 首次运行很慢?

A: 首次运行会自动下载预训练模型和数据集,需要一些时间。模型会缓存到 `~/.cache/torch/hub/checkpoints/`。

### Q2: CUDA out of memory?

A: 减小batch_size或使用CPU模式:

```python
model = load_model('resnet50', device='cpu')
```

### Q3: 如何使用自己的图像?

A: 在Web界面选择"自定义图像",然后上传PNG/JPG格式的图像。

### Q4: 如何添加新的攻击算法?

A: 在 `src/attacks/` 目录下创建新文件,继承 `BaseAttack` 类:

```python
from .base import BaseAttack

class MyAttack(BaseAttack):
    def generate(self, images, labels, **kwargs):
        # 实现你的攻击算法
        pass
```

## 项目结构

```
multimodal-attack-platform/
├── src/                    # 源代码
│   ├── attacks/           # 攻击算法
│   ├── models/            # 模型管理
│   ├── evaluation/        # 评估指标
│   ├── data_manager/      # 数据管理
│   └── visualization/     # 可视化
├── web/                   # Web界面
├── data/                  # 数据目录
├── docs/                  # 文档
├── config/                # 配置文件
├── demo.py               # 示例脚本
├── requirements.txt      # 依赖包
└── README.md            # 项目说明
```

## 下一步

- 阅读 [算法说明文档](docs/算法说明.md) 了解攻击原理
- 阅读 [总结报告](docs/总结报告.md) 查看实验结果
- 尝试不同的模型和攻击参数
- 使用自己的数据集进行实验

## 获取帮助

如有问题,请查看:
- 文档目录下的详细说明
- 代码注释
- 示例脚本

祝实验顺利! 🚀
