# 安装与运行指南

## 环境要求

- Python 3.10+
- 推荐使用 Anaconda 环境
- CUDA 可选；有 CUDA 时生成对抗样本更快
- 阿里云百炼 API Key，用于 Qwen-VL 黑盒评估

## 1. 安装依赖

如果已有课程项目环境：

```cmd
conda activate multimodal-attack
pip install -r requirements.txt
```

如果需要新建环境：

```cmd
conda create -n multimodal-attack python=3.10 -y
conda activate multimodal-attack
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

没有 CUDA 时可安装 CPU 版：

```cmd
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

## 2. 配置 Qwen-VL API

华北2（北京）地域：

```cmd
set DASHSCOPE_API_KEY=你的百炼APIKey
set DASHSCOPE_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
```

新加坡地域：

```cmd
set DASHSCOPE_API_KEY=你的百炼APIKey
set DASHSCOPE_BASE_URL=https://dashscope-intl.aliyuncs.com/compatible-mode/v1
```

美国（弗吉尼亚）地域：

```cmd
set DASHSCOPE_API_KEY=你的百炼APIKey
set DASHSCOPE_BASE_URL=https://dashscope-us.aliyuncs.com/compatible-mode/v1
```

API Key 不要写入代码，不要提交到 GitHub。

如果 Key 曾经发到聊天、截图或公开仓库，建议在阿里云控制台重新生成 Key，并删除旧 Key。

## 3. 启动 Web 平台

```cmd
cd "D:\desk\作业\大三下\未来互联网新技术\Network_lab"
conda activate multimodal-attack
streamlit run web/app.py
```

浏览器访问：

```text
http://localhost:8501
```

## 4. 验证环境

```cmd
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
python test_imports.py
```

## 5. 缓存和费用控制

黑盒评估默认使用本地缓存：

```text
data/blackbox_cache/
```

同一张图片、同一模型、同一提示词不会重复调用 API。若要强制重新请求，可修改提示词末尾，例如加 `版本2`。

推荐低成本模型：

```text
qwen-vl-plus-latest
```

OCR/表格场景优先：

```text
qwen-vl-ocr
```

少量高质量补充：

```text
qwen-vl-max-latest
```

## 6. 常见问题

### 自定义图片下分类准确率为什么是 0？

自定义截图不是 ImageNet 分类任务，ResNet 的分类准确率没有实验意义。该页只用于生成扰动图。最终结论看 `Qwen-VL 黑盒评估` 页的字段准确率和真实答案攻击成功。

### epsilon=0 时应该怎样？

`epsilon=0` 是对照组。应看到：

```text
L2=0
Linf=0
SSIM=1
真实答案攻击成功=否
```

### 原图也识别错怎么办？

该样本不能算严格攻击成功，只能作为“原图识别困难样本”。严格成功样本要求原图字段准确率至少 80%。
