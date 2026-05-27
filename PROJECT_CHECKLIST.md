# 项目交付清单与当前状态

## 当前定位

项目已从“本地 CLIP/BLIP 多模态攻击演示”收口为：

```text
Qwen-VL 图像对抗鲁棒性评估平台
```

当前 GitHub 交接入口：

```text
README.md
QUICKSTART.md
EXPERIMENT_GUIDE.md
TEAM_HANDOFF.md
IMPROVEMENTS.md
REPORT_PPT_GUIDE.md
```

主线是：

```text
样本管理
-> 对抗样本生成
-> Qwen-VL 黑盒字段评估
-> 鲁棒性分析
-> 结果可视化与实验记录
```

## 指导书要求对照

| 指导书要求 | 当前状态 | 证据 |
| --- | --- | --- |
| 数据样本管理 | 已实现 | `src/data_manager/dataset_manager.py`，Web `样本管理` 页 |
| 图像对抗样本生成 | 已实现 | `src/attacks/fgsm.py`、`pgd.py`、`cw.py`，Web `对抗样本生成` 页 |
| 攻击效果评估 | 已实现 | L2、Linf、SSIM、本地 surrogate 预测翻转、Qwen-VL 字段准确率 |
| 鲁棒性分析 | 已实现 | 原图/对抗图字段准确率对比、真实答案攻击成功判定、失败样本分析 |
| 结果可视化与实验记录 | 已实现 | 原图/对抗图/扰动图展示，`data/adversarial/` metadata |
| 黑盒迁移评测 | 已实现 | `src/external_models/vision_clients.py` + `src/evaluation/blackbox_metrics.py` |
| 主流大模型 | 已实现 | 阿里云百炼 Qwen-VL，默认 `qwen-vl-plus-latest`，支持 `qwen-vl-ocr` |
| 实验报告/PPT 支撑 | 已整理 | `REPORT_PPT_GUIDE.md`、`EXPERIMENT_GUIDE.md`、`docs/sample_images/` |

## 本次主要改进

### 1. 主线 UI 重构

- 页面标题改为 `Qwen-VL 图像对抗鲁棒性评估平台`。
- Tab 顺序调整为：

```text
对抗样本生成
Qwen-VL 黑盒评估
实验结果分析
样本管理
历史记录
附加实验
使用说明
```

- CLIP/BLIP 被移动到 `附加实验`，不再干扰主线。

### 2. Qwen-VL 黑盒评估

新增：

- DashScope / Qwen-VL OpenAI-compatible API 调用。
- `DASHSCOPE_API_KEY`、`DASHSCOPE_BASE_URL` 环境变量配置。
- 黑盒调用本地缓存 `data/blackbox_cache/`，避免重复扣费。
- Qwen 模型选择：

```text
qwen-vl-plus-latest
qwen-vl-max-latest
qwen-vl-ocr
```

### 3. 真实答案字段评估

新增指标：

- 回答相似度
- 输出变化
- 字段变化数
- 数字变化数
- 原图字段准确率
- 对抗图字段准确率
- 真实答案攻击成功

严格攻击成功标准：

```text
原图字段准确率 >= 80%
对抗图字段准确率下降
真实答案攻击成功 = 是
```

### 4. 自定义图片分类指标清理

自定义截图不是 ImageNet 分类任务，因此隐藏无意义的：

```text
原始准确率
攻击后准确率
攻击成功率
```

保留：

```text
Surrogate 预测翻转
L2 扰动
Linf 扰动
SSIM
```

最终成功与否以 Qwen-VL 黑盒字段评估为准。

## 推荐实验材料

优先使用：

- 课程通知截图
- 菜单/价目表
- 天气信息截图
- 简单成绩表
- 快递/订单信息截图

不推荐作为主成功样本：

- 猫狗等自然图像，通常太容易识别。
- 过于复杂的 App 截图，原图可能本来就识别错误。
- 原图字段准确率低于 60% 的截图。

## 已验证样本类型

| 样本类型 | 现象 | 报告用途 |
| --- | --- | --- |
| 课程通知截图 | 原图字段准确率较高，对抗图字段准确率下降 | 主成功样本 |
| 简单成绩表 | 原图和对抗图均可正确识别 | 鲁棒失败样本 |
| 体育 App 截图 | 原图本身存在误读 | 原图识别困难样本 |
| 猫/动漫图 | 输出较稳定，难以迁移成功 | 可选失败样本 |

## 仍需队友补充

1. 至少收集 5 张测试截图。
2. 每张图跑 `epsilon=0`、`epsilon=0.03`、`epsilon=0.05` 三组。
3. 记录字段准确率、真实答案攻击成功、L2/Linf/SSIM。
4. 截取关键页面用于报告和 PPT。
5. 将 API Key 从截图和提交文件中删除。
6. 把最终结果填入 `EXPERIMENT_GUIDE.md` 的结果表模板或复制到报告中。

## 不应写进报告的误区

- 不要把自定义图片下的 ResNet 分类准确率 0% 当成失败。
- 不要把“回答格式变化”直接写成攻击成功。
- 不要把原图本来识别错的样本计入严格攻击成功率。
- 不要声称已经对白盒攻击 Qwen-VL；商业模型只能做黑盒评估。

## 当前完成度判断

功能层面：已满足指导书核心要求。  
报告层面：还需要队友补充多组实验截图和统计表。  
风险点：黑盒 API 调用费用、模型输出随机性、测试图选择会影响成功率。
