# 本次改进清单

这份文件给队友和报告/PPT 撰写者快速了解：本次项目到底从哪里改到了哪里。

## 1. 项目主线调整

旧状态：

```text
本地分类攻击、CLIP/BLIP 多模态演示较多，商业大模型评估不清晰。
```

当前状态：

```text
对抗样本生成 -> Qwen-VL 黑盒字段评估 -> 实验结果分析 -> 历史记录
```

答辩主线已经明确为：

```text
面向 Qwen-VL 的图像对抗样本黑盒鲁棒性评估
```

## 2. 前端改进

主要文件：

```text
web/app.py
```

改进内容：

- 页面标题改为 `Qwen-VL 图像对抗鲁棒性评估平台`。
- Tab 顺序改为主线优先：对抗样本生成、Qwen-VL 黑盒评估、实验结果分析、样本管理、历史记录。
- CLIP/BLIP 移到 `附加实验`，不再作为答辩主流程。
- 侧边栏主线参数只保留生成 Qwen-VL 评估所需对抗图的配置。
- Qwen-VL 页只暴露 Alibaba Qwen-VL，避免 OpenAI/Gemini/Claude 干扰。
- 原图字段准确率低于 80% 时，页面会提示该样本不能计入严格攻击成功。

## 3. 商业模型黑盒评估

主要文件：

```text
src/external_models/vision_clients.py
src/evaluation/blackbox_metrics.py
src/evaluation/__init__.py
```

实现内容：

- 接入阿里云百炼 DashScope OpenAI-compatible API。
- 支持 `DASHSCOPE_API_KEY` 和 `DASHSCOPE_BASE_URL` 环境变量。
- 支持 `qwen-vl-plus-latest`、`qwen-vl-max-latest`、`qwen-vl-ocr`。
- 启用 `data/blackbox_cache/` 本地缓存，减少重复调用费用。
- 对比原图和对抗图回答，计算回答相似度、字段变化数、数字变化数。
- 支持真实答案字段 `key=value`，计算原图字段准确率和对抗图字段准确率。

## 4. 成功判定改进

旧问题：

```text
如果原图本来就识别错，容易被误看成攻击成功。
```

当前判定：

```text
原图字段准确率 >= 80%
对抗图字段准确率下降
存在原本正确但对抗后错误的字段
L2/Linf > 0
```

因此，体育 App 截图这类原图识别困难样本不能算严格成功，只能作为复杂场景分析。

## 5. 本地评估修复

主要文件：

```text
src/evaluation/metrics.py
```

修复内容：

- 修复 `target_labels` 和预测张量不在同一设备时的报错。
- 自定义截图场景下不再把 ResNet 分类准确率作为最终结论。

## 6. 文档改进

新增或重写：

```text
README.md
QUICKSTART.md
INSTALL.md
PROJECT_CHECKLIST.md
TEAM_HANDOFF.md
REPORT_PPT_GUIDE.md
EXPERIMENT_GUIDE.md
docs/总结报告.md
docs/算法说明.md
material-checklist.md
project-documentation-outline.md
self-check-report.md
```

文档现在覆盖：

- 项目定位和指导书要求对应。
- 启动命令和 API 配置。
- 推荐测试图、参数、提示词、真实答案字段。
- 报告/PPT 结构、结果表和结论模板。
- GitHub 上传前的 Key、缓存、截图注意事项。

## 7. 示例实验素材

示例图片目录：

```text
docs/sample_images/
```

包含：

```text
课程通知截图.png
生成菜单截图.png
生成成绩表截图.png
北京天气截图.png
```

建议用途：

- `课程通知截图.png`：优先寻找严格成功样本。
- `生成菜单截图.png`：观察价格数字是否被扰动影响。
- `生成成绩表截图.png`：常用于鲁棒失败样本。
- `北京天气截图.png`：字段少，适合补充和对照。

## 8. GitHub 上传注意事项

不要提交：

```text
data/
.env
*.env
*.key
secrets.*
API Key 截图
```

可以提交：

```text
代码
README 和各类指南
docs/sample_images/ 中的无敏感示例图片
```

如果 API Key 已经出现在聊天、截图或公开仓库中，建议到阿里云控制台重新生成。

## 9. 队友还需要补充

- 至少 5 张测试图。
- 每张图跑 `epsilon=0`、`0.03`、`0.05` 三组。
- 记录 Qwen-VL 原图字段准确率、对抗图字段准确率、真实答案攻击成功、L2/Linf/SSIM。
- 截取对抗样本生成页、Qwen-VL 评估页、回答对比、历史记录页。
- 把最终结果填入 `EXPERIMENT_GUIDE.md` 的表格，再写进报告和 PPT。
