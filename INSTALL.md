# 安装与运行指南

## 环境要求

- Python 3.10+
- Windows / Linux / macOS
- CUDA 可选，但建议用于加速实验

## 1. 安装 PyTorch

### CPU 版本

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

### CUDA 版本

请根据你的本机 CUDA 版本安装对应的 PyTorch 版本，例如：

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

## 2. 安装项目依赖

```bash
pip install -r requirements.txt
```

## 3. 验证环境

```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

## 4. 启动 Web 平台

```bash
streamlit run web/app.py
```

## 5. 首次运行说明

当前默认主流程直接面向 `CLIP / BLIP` 多模态目标，不依赖 `CIFAR-10 / MNIST` 分类权重。

首次点击对应评测或攻击入口时，`transformers` 会自动下载所需模型到本地缓存。

样本上传、标注、版本快照和 CSV/JSON 导出不需要额外依赖。

## 6. 可选的分类基线

如果你仍然需要运行旧的分类基线实验，可额外训练对应权重：

```bash
python scripts/train_classifier.py --dataset cifar10 --model resnet18 --epochs 5
python scripts/train_classifier.py --dataset mnist --model resnet18 --epochs 3
```

## 7. 常见问题

### Q1：程序提示首次加载模型较慢

说明当前正在下载或读取 `CLIP / BLIP` 本地缓存，属于正常现象。

### Q2：程序只能检测到 CPU

说明当前 Python 环境安装的是 CPU 版 PyTorch，或 CUDA 环境未正确配置。

### Q3：旧分类基线提示没有匹配权重

说明你进入的是附录性质的分类实验路径，但还没有训练对应权重。请先执行上面的训练脚本。

### Q4：为什么“攻击成功”不是只看描述是否变化

当前平台已经为 CLIP / BLIP 固化了统一判定规则。结果页会同时展示阈值判定、关键指标和原因说明。
