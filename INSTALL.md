# 安装与运行指南

## 环境要求

- Python 3.8+
- CUDA（可选，用于 GPU 加速）

## 安装步骤

### 1. 安装 PyTorch

**CPU 版本：**
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

**GPU 版本（CUDA 11.8）：**
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

**GPU 版本（CUDA 12.1）：**
```bash
pip install torch torchvision
```

### 2. 安装其他依赖

```bash
pip install -r requirements.txt
```

或者手动安装：
```bash
pip install numpy pillow matplotlib plotly streamlit scikit-learn scipy pandas opencv-python tqdm tensorboard seaborn transformers timm
```

## 运行方式

### 方式 1：命令行 Demo

```bash
python demo.py
```

这将：
- 加载 ResNet-50 模型
- 使用 CIFAR-10 数据集
- 执行 PGD 攻击（100 个样本）
- 评估攻击效果
- 保存结果到 `data/adversarial/`
- 生成可视化图表

### 方式 2：Web 界面（推荐）

```bash
streamlit run web/app.py
```

然后在浏览器打开 `http://localhost:8501`

Web 界面功能：
- 📊 攻击实验：配置参数、执行攻击、查看结果
- 📈 结果分析：查看详细指标和可视化
- 📁 实验记录：浏览历史实验
- 📖 使用说明：查看帮助文档

### 方式 3：Python API

```python
from src.attacks import FGSM, PGD, CarliniWagner
from src.models import load_model
from src.data_manager import DatasetManager
from src.evaluation import AttackEvaluator

# 加载模型
model = load_model('resnet50', pretrained=True, num_classes=10, device='cuda')

# 加载数据
data_manager = DatasetManager()
dataloader = data_manager.load_dataset('cifar10', split='test', batch_size=32)

# 创建攻击器
attacker = PGD(model, epsilon=0.03, alpha=0.01, num_iter=10, device='cuda')

# 执行攻击
for images, labels in dataloader:
    adv_images, info = attacker.generate(images, labels)
    break

# 评估
evaluator = AttackEvaluator(model, device='cuda')
metrics = evaluator.evaluate(images, adv_images, labels)
print(f"攻击成功率: {metrics['attack_success_rate']:.2f}%")
```

## 常见问题

### Q1: 提示 "No module named 'torch'"
A: 需要先安装 PyTorch，参考上面的安装步骤。

### Q2: CUDA out of memory
A: 减小 batch_size 或使用 CPU 模式（device='cpu'）。

### Q3: 下载数据集很慢
A: CIFAR-10 和 MNIST 会自动下载，首次运行需要等待。可以手动下载后放到 `data/raw/` 目录。

### Q4: Web 界面无法访问
A: 确保 8501 端口未被占用，或使用 `streamlit run web/app.py --server.port 8502` 指定其他端口。

## 项目结构

```
Network_lib1/
├── src/                    # 源代码
│   ├── attacks/           # 攻击算法（FGSM、PGD、C&W）
│   ├── models/            # 模型加载
│   ├── data_manager/      # 数据集管理
│   ├── evaluation/        # 评估指标
│   └── visualization/     # 可视化
├── web/                   # Web 界面
│   └── app.py
├── data/                  # 数据目录
│   ├── raw/              # 原始数据集
│   ├── processed/        # 处理后数据
│   └── adversarial/      # 对抗样本
├── docs/                  # 文档
│   ├── 算法说明.md
│   └── 总结报告.md
├── config/                # 配置文件
│   └── config.yaml
├── demo.py               # 命令行示例
├── requirements.txt      # 依赖列表
├── README.md            # 项目说明
└── QUICKSTART.md        # 快速开始
```

## 支持的模型

- ResNet-18/50/101
- VGG-16/19
- DenseNet-121
- MobileNet-V2
- EfficientNet-B0

## 支持的数据集

- CIFAR-10（自动下载）
- MNIST（自动下载）
- ImageNet（需手动准备）
- 自定义图像（Web 界面上传）

## 支持的攻击算法

- **FGSM**：快速梯度符号方法
- **PGD**：投影梯度下降
- **C&W**：Carlini & Wagner L2 攻击

## 评估指标

- 攻击成功率（ASR）
- 扰动强度（L0/L2/L∞）
- 感知质量（SSIM/PSNR）
- 置信度变化
- 迁移性测试
