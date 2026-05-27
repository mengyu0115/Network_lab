# 面向主流多模态大模型的图像对抗鲁棒性评估平台

## 项目定位

本项目面向课程项目场景，围绕图片模态构建对抗样本生成、攻击效果评估、黑盒迁移鲁棒性分析和结果可视化平台。

平台不研究隐藏载荷投递、绕过平台安全策略或真实系统滥用。项目采用合规的安全评估口径：先用本地 surrogate 模型生成图像对抗样本，再通过授权 API 对阿里云百炼 Qwen-VL 等主流多模态大模型进行黑盒输出对比。

## 给队友的阅读顺序

如果是第一次接手项目，建议按这个顺序看：

1. `QUICKSTART.md`：启动平台并跑通一组实验。
2. `EXPERIMENT_GUIDE.md`：选择测试图、设置参数、填写提示词和真实答案。
3. `TEAM_HANDOFF.md`：了解当前改进、主线页面和注意事项。
4. `IMPROVEMENTS.md`：查看本次代码和文档具体改了什么。
5. `REPORT_PPT_GUIDE.md`：直接写实验报告和 PPT。
6. `PROJECT_CHECKLIST.md`：核对指导书要求和当前完成度。

本地过程记录、课程原始材料和临时截图不作为 GitHub 交付内容；当前主线以以上文档为准。

## 指导书要求对应

| 指导书要求 | 实现方式 |
| --- | --- |
| 数据样本管理 | 图片上传、标注、分组、版本快照、CSV/JSON 导出 |
| 图像对抗样本生成 | FGSM、PGD、C&W 生成原图/对抗图/扰动图 |
| 攻击效果评估 | 攻击成功率、预测变化、L2/Linf、SSIM、PSNR |
| 鲁棒性分析 | Qwen-VL 黑盒字段准确率下降、真实答案攻击成功、失败样本分析 |
| 结果可视化与实验记录 | Streamlit 页面、指标表格、图片对比、历史记录、metadata 保存 |

## 核心功能

1. 对抗样本生成
   - 支持自定义图片、CIFAR-10、MNIST。
   - 支持 FGSM、PGD、C&W。
   - 使用本地 surrogate 模型生成原图、对抗图、扰动图。
   - 自定义截图场景下不使用 ImageNet 分类准确率作为最终结论。

2. 多模态 surrogate 攻击
   - 支持 CLIP 图文对齐攻击。
   - 支持 BLIP 图像描述扰动验证。
   - 用于生成和分析可迁移的图片扰动。

3. Qwen-VL 黑盒评估
   - 默认支持阿里云百炼 Qwen-VL 的授权 API 调用。
   - 支持主体识别、自由描述、文档/OCR、目标误导四类任务化评估。
   - 可先识别原图主体，再检测对抗图回答是否丢失该主体。
   - 可填写真实答案字段，区分“输出发生变化”和“原图正确但对抗图错误”的真正攻击成功。
   - 输出回答相似度、关键词丢失、字段准确率、目标关键词命中、黑盒迁移成功率等指标。

4. 多模态本地攻击附加实验
   - 保留 CLIP / BLIP 作为本地 baseline 和扩展实验。
   - 答辩主线以 Qwen-VL 黑盒评估为准，CLIP/BLIP 不作为市面商业大模型结论。

5. 样本和实验管理
   - 支持图片样本上传、标注、分组和版本管理。
   - 支持历史实验回放。
   - 支持实验索引导出。

## 技术路线

```text
图片样本
  -> 本地模型生成对抗图（FGSM / PGD / C&W）
  -> 本地评估（分类模型 / CLIP / BLIP）
  -> Qwen-VL 黑盒字段评估
  -> 指标统计、可视化和实验记录
```

商业模型无法获取梯度，因此本项目采用“白盒 surrogate 生成 + 黑盒迁移验证”的方式评估市面主流多模态大模型鲁棒性。

## 项目结构

```text
Network_lab/
├── src/
│   ├── attacks/              # FGSM / PGD / C&W / CLIP / BLIP 攻击实现
│   ├── data_manager/         # 样本管理、版本快照、实验记录
│   ├── evaluation/           # 攻击指标、CLIP/BLIP 评估、黑盒输出指标
│   ├── external_models/      # Qwen-VL/DashScope 黑盒客户端与缓存
│   ├── models/               # 本地模型加载
│   └── visualization/        # 可视化工具
├── web/
│   └── app.py                # Streamlit Web 平台
├── scripts/
│   └── train_classifier.py   # 分类模型微调脚本
├── data/
│   ├── raw/                  # 原始样本
│   ├── processed/            # 处理后样本和版本
│   └── adversarial/          # 对抗实验记录，默认不提交 GitHub
├── docs/
│   ├── sample_images/        # 推荐测试截图素材
│   ├── 总结报告.md
│   └── 算法说明.md
├── config/
├── requirements.txt
├── EXPERIMENT_GUIDE.md       # 实验样本、参数、提示词与结果表模板
├── TEAM_HANDOFF.md           # 队友交接说明
├── REPORT_PPT_GUIDE.md       # 报告与 PPT 写作指南
└── README.md
```

## 安装运行

安装依赖：

```bash
pip install -r requirements.txt
```

启动 Web 平台：

```bash
streamlit run web/app.py
```

默认访问地址：

```text
http://localhost:8501
```

## 商业模型 API 配置

如需使用 `Qwen-VL 黑盒评估` 页面，先配置阿里云百炼 Key：

```bash
set DASHSCOPE_API_KEY=your_dashscope_key
```

默认模型为：

```text
qwen-vl-plus-latest
```

如果需要更强但更贵的补充实验，可以在页面里手动改为：

```text
qwen-vl-max-latest
```

API Key 不会写入实验记录。平台默认启用 `data/blackbox_cache/` 本地缓存，同一图片、同一模型、同一提示词不会重复调用 API。

## 推荐演示流程

1. 在 `对抗样本生成` 页上传一张图片。
2. 选择 surrogate 模型和攻击方法，例如 PGD 或 C&W。
3. 生成对抗图并查看原图、对抗图、扰动图。
4. 在“结果分析”页查看扰动强度、攻击成功率和本地迁移评测。
5. 在“Qwen-VL 黑盒评估”页选择 Alibaba Qwen-VL。
6. 选择“主体识别”或“文档/OCR”等评估任务。
7. 可先点击“先识别原图主体并填入保留关键词”，再运行黑盒评估。
8. 对文档/OCR 场景建议填写真实答案字段，例如 `得分=41`、`篮板=24`，再运行黑盒评估。
9. 对比商业模型对原图和对抗图的回答，查看字段准确率下降和迁移成功判定。
10. 在“历史记录”页回看保存的实验配置和指标。

推荐先使用 `docs/sample_images/课程通知截图.png` 或 `docs/sample_images/生成菜单截图.png`。具体提示词、真实答案字段和结果记录表见 `EXPERIMENT_GUIDE.md`。

## 合规说明

本项目仅用于课程实验和模型安全评估。实验输入为用户可控图片，对商业模型的调用基于公开 API 和授权 Key，不包含隐藏载荷投递、绕过安全策略、自动化滥用或真实系统攻击流程。

## 已知限制

- 商业闭源模型只能进行黑盒输出评估，无法直接执行白盒梯度攻击。
- 商业 API 调用需要网络、Key 和可能的费用；建议主实验使用 `qwen-vl-plus-latest`，少量补充使用 `qwen-vl-max-latest`。
- 标准数据集分类攻击需要对应微调权重，否则实验会提示先训练；主线建议使用自定义截图。
- 当前黑盒指标基于任务化提示、真实答案字段、文本输出变化和关键词命中；报告中应区分“输出变化”和“真实答案意义上的攻击成功”。
- `data/`、API 缓存、`.env`、Key 文件默认不应上传 GitHub；如终端截图里出现 Key，需要打码或重新生成 Key。
