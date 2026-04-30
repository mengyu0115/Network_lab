"""
图像对抗攻击平台 - Streamlit 页面
"""
import os
import sys
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
import streamlit as st
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.attacks import (
    FGSM,
    PGD,
    CarliniWagner,
)
from src.data_manager import DatasetManager
from src.evaluation import AttackEvaluator
from src.models import ModelLoader, load_model


def to_perturbation_vis(perturbation: np.ndarray) -> np.ndarray:
    max_val = float(perturbation.max())
    if max_val <= 1e-12:
        return perturbation
    return np.clip(perturbation / max_val, 0.0, 1.0)


@st.cache_resource(show_spinner=False)
def get_cached_vision_model(
    model_name: str,
    device: str,
    num_classes: int = 1000,
    dataset_name: str = "ImageNet",
    checkpoint_version: str = "",
):
    return load_model(
        model_name,
        pretrained=True,
        device=device,
        num_classes=num_classes,
        dataset_name=dataset_name,
    )


st.set_page_config(
    page_title="图像对抗攻击平台",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("🛡️ 图像对抗攻击平台")
st.markdown("---")

st.sidebar.header("配置")
attack_method = st.sidebar.selectbox(
    "攻击方法",
    ["FGSM", "PGD", "C&W"],
    index=0,
)
vision_models = ModelLoader.get_supported_models(include_clip=False, include_caption=False)
model_name = st.sidebar.selectbox("目标模型", vision_models, index=1 if len(vision_models) > 1 else 0)
dataset_name = st.sidebar.selectbox("数据集", ["CIFAR-10", "MNIST", "自定义图片"], index=0)

st.sidebar.subheader("攻击参数")
if attack_method == "FGSM":
    attack_params = {"epsilon": st.sidebar.slider("epsilon", 0.0, 0.3, 0.03, 0.01)}
elif attack_method == "PGD":
    attack_params = {
        "epsilon": st.sidebar.slider("epsilon", 0.0, 0.3, 0.03, 0.01),
        "alpha": st.sidebar.slider("步长(alpha)", 0.001, 0.05, 0.01, 0.001),
        "num_iter": st.sidebar.slider("迭代次数", 5, 50, 10, 5),
    }
elif attack_method == "C&W":
    attack_params = {
        "c": st.sidebar.slider("损失权重(c)", 0.1, 10.0, 1.0, 0.1),
        "learning_rate": st.sidebar.slider("学习率", 0.001, 0.1, 0.01, 0.001),
        "num_iter": st.sidebar.slider("迭代次数", 20, 200, 100, 10),
    }

targeted = st.sidebar.checkbox("目标攻击", value=False)
device = "cuda" if torch.cuda.is_available() else "cpu"
st.sidebar.info(f"设备：{device.upper()}")

tab1, tab2, tab3, tab4 = st.tabs(["攻击实验", "结果分析", "历史记录", "使用说明"])

with tab1:
    st.header("运行攻击")
    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("输入")
        uploaded_file = None
        image_tensor = None
        true_label = 0
        target_label = None
        num_samples = 10

        if dataset_name == "自定义图片":
            uploaded_file = st.file_uploader("上传图片", type=["png", "jpg", "jpeg"], key="img_upload")
            if uploaded_file is not None:
                image = Image.open(uploaded_file).convert("RGB")
                st.image(image, caption="上传图片", use_container_width=True)
                from torchvision import transforms

                transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
                image_tensor = transform(image).unsqueeze(0)
                true_label = st.number_input("真实标签", min_value=0, max_value=999, value=0)
                if targeted:
                    target_label = st.number_input("目标标签", min_value=0, max_value=999, value=1)
        else:
            st.info("标准数据集批量测试")
            num_samples = st.slider("样本数", 1, 100, 10)

    with col2:
        st.subheader("执行")
        if st.button("开始图像攻击", type="primary", use_container_width=True):
            if dataset_name == "自定义图片" and (uploaded_file is None or image_tensor is None):
                st.error("请先上传一张图片。")
                st.stop()

            with st.spinner("加载模型（首次会下载，后续复用缓存）..."):
                dataset_num_classes = {"CIFAR-10": 10, "MNIST": 10}.get(dataset_name, 1000)
                checkpoint_path = ModelLoader.get_finetune_checkpoint_path(model_name, dataset_name)
                checkpoint_version = (
                    str(os.path.getmtime(checkpoint_path))
                    if checkpoint_path and os.path.exists(checkpoint_path)
                    else "missing"
                )
                model = get_cached_vision_model(
                    model_name,
                    device,
                    dataset_num_classes,
                    dataset_name,
                    checkpoint_version,
                )
                if dataset_num_classes != 1000:
                    if getattr(model, "fine_tuned_checkpoint_loaded", False):
                        ckpt = str(getattr(model, "fine_tuned_checkpoint_path", ""))
                        st.success(f"已自动加载 {dataset_name} 微调权重：{ckpt}")
                    else:
                        ckpt_status = str(getattr(model, "fine_tuned_checkpoint_status", ""))
                        if ckpt_status == "checkpoint_not_found":
                            ckpt = str(getattr(model, "fine_tuned_checkpoint_path", ""))
                            st.error(
                                f"已将分类头改为 {dataset_num_classes} 类以匹配 {dataset_name}，但未找到对应微调权重：{ckpt}。"
                                " 标准数据集实验会被停止，避免生成不可信结果。"
                            )
                            st.code(
                                f"python scripts/train_classifier.py --dataset {dataset_name.lower().replace('-', '')} "
                                f"--model {model_name} --epochs 5",
                                language="bash",
                            )
                            st.stop()
                        else:
                            st.error(
                                f"已将分类头改为 {dataset_num_classes} 类以匹配 {dataset_name}。"
                                " 该分类头未在该数据集微调，标准数据集实验会被停止。"
                            )
                            st.stop()

            if attack_method == "FGSM":
                attacker = FGSM(model, **attack_params, device=device)
            elif attack_method == "PGD":
                attacker = PGD(model, **attack_params, device=device)
            else:
                attacker = CarliniWagner(model, **attack_params, device=device)
            data_manager = DatasetManager()
            attack_name = attack_method.replace("&", "and").replace("/", "_")

            if dataset_name == "自定义图片" and uploaded_file is not None and image_tensor is not None:
                labels = torch.tensor([int(true_label)])
                target_tensor = torch.tensor([int(target_label)]) if targeted and target_label is not None else None
                adv_images, info = attacker.generate(image_tensor, labels, targeted=targeted, target_labels=target_tensor)
                orig_display = image_tensor[0].detach().cpu()
                adv_display = adv_images[0].detach().cpu()

                with torch.no_grad():
                    orig_pred = model(image_tensor.to(device)).argmax(dim=1).item()
                    adv_pred = model(adv_images.to(device)).argmax(dim=1).item()

                st.success("对抗样本生成完成。")
                c1, c2, c3 = st.columns(3)
                with c1:
                    st.image(image, caption=f"原图(预测={orig_pred})", use_container_width=True)
                with c2:
                    adv_img = adv_display.permute(1, 2, 0).numpy()
                    st.image(np.clip(adv_img, 0, 1), caption=f"对抗图(预测={adv_pred})", use_container_width=True)
                with c3:
                    pert = (adv_display - orig_display).abs().permute(1, 2, 0).numpy()
                    st.image(to_perturbation_vis(pert), caption="扰动(归一化)", use_container_width=True)

                st.metric("攻击成功", "是" if orig_pred != adv_pred else "否")
                st.write(f"L2扰动：{info['perturbation_l2']:.4f}")
                st.write(f"Linf扰动：{info['perturbation_linf']:.4f}")

                evaluator = AttackEvaluator(model, device=device)
                metrics = evaluator.evaluate(
                    image_tensor.cpu(),
                    adv_images.detach().cpu(),
                    labels.cpu(),
                    targeted=targeted,
                    target_labels=target_tensor.cpu() if target_tensor is not None else None,
                )
                save_dir = data_manager.save_adversarial_samples(
                    image_tensor.detach().cpu(),
                    labels.cpu(),
                    adv_images.detach().cpu(),
                    torch.tensor([int(adv_pred)]),
                    attack_name,
                    metadata={
                        "model": model_name,
                        "dataset": dataset_name,
                        "targeted": targeted,
                        "target_label": int(target_label) if target_label is not None else None,
                        "metrics": {k: float(v) for k, v in metrics.items() if isinstance(v, (int, float, np.floating, np.integer))},
                    },
                )
                st.info(f"实验已保存到：{save_dir}")
            else:
                dataloader = data_manager.load_dataset(dataset_name.lower().replace("-", ""), split="test", batch_size=32, shuffle=False)

                all_original, all_adversarial, all_labels = [], [], []
                all_orig_preds, all_adv_preds = [], []
                sample_count = 0
                output_dim = None
                pb = st.progress(0)
                stat = st.empty()

                for images, labels in dataloader:
                    if sample_count >= num_samples:
                        break
                    remaining = num_samples - sample_count
                    if remaining < len(images):
                        images = images[:remaining]
                        labels = labels[:remaining]
                    images = images.to(device)
                    labels = labels.to(device)
                    adv_images, _ = attacker.generate(images, labels, targeted=targeted)

                    with torch.no_grad():
                        orig_outputs = model(images)
                        adv_outputs = model(adv_images)
                        if output_dim is None:
                            output_dim = int(orig_outputs.shape[1])
                        orig_preds = orig_outputs.argmax(dim=1)
                        adv_preds = adv_outputs.argmax(dim=1)

                    all_original.append(images.cpu())
                    all_adversarial.append(adv_images.cpu())
                    all_labels.append(labels.cpu())
                    all_orig_preds.append(orig_preds.cpu())
                    all_adv_preds.append(adv_preds.cpu())

                    sample_count += len(images)
                    pb.progress(min(sample_count / num_samples, 1.0))
                    stat.text(f"已处理 {sample_count}/{num_samples} 个样本")

                all_original = torch.cat(all_original)[:num_samples]
                all_adversarial = torch.cat(all_adversarial)[:num_samples]
                all_labels = torch.cat(all_labels)[:num_samples]
                all_orig_preds = torch.cat(all_orig_preds)[:num_samples]
                all_adv_preds = torch.cat(all_adv_preds)[:num_samples]

                evaluator = AttackEvaluator(model, device=device)
                metrics = evaluator.evaluate(all_original, all_adversarial, all_labels)
                st.success(f"已生成 {num_samples} 个对抗样本。")
                save_dir = data_manager.save_adversarial_samples(
                    all_original,
                    all_labels,
                    all_adversarial,
                    all_adv_preds,
                    attack_name,
                    metadata={
                        "model": model_name,
                        "dataset": dataset_name,
                        "targeted": targeted,
                        "num_samples": int(num_samples),
                        "metrics": {k: float(v) for k, v in metrics.items() if isinstance(v, (int, float, np.floating, np.integer))},
                    },
                )
                st.info(f"实验已保存到：{save_dir}")

                expected_num_classes = {"CIFAR-10": 10, "MNIST": 10}.get(dataset_name)
                label_space_mismatch = expected_num_classes is not None and output_dim is not None and output_dim != expected_num_classes
                if label_space_mismatch:
                    st.warning(
                        f"模型输出类别数 {output_dim} 与数据集类别数 {expected_num_classes} 不匹配。"
                        " 当前结果应解读为预测翻转强度，不是标准分类准确率。"
                    )

                st.session_state["last_experiment"] = {
                    "model": model_name,
                    "dataset": dataset_name,
                    "attack": attack_method,
                    "metrics": metrics,
                    "num_samples": num_samples,
                    "original": all_original[:10],
                    "adversarial": all_adversarial[:10],
                    "labels": all_labels[:10],
                    "orig_preds": all_orig_preds[:10],
                    "adv_preds": all_adv_preds[:10],
                    "label_space_mismatch": label_space_mismatch,
                    "model_output_dim": output_dim,
                }

                c1, c2, c3 = st.columns(3)
                with c1:
                    st.metric("原始准确率", f"{metrics['original_accuracy']:.2f}%")
                with c2:
                    st.metric("攻击后准确率", f"{metrics['adversarial_accuracy']:.2f}%")
                with c3:
                    st.metric("预测翻转率", f"{metrics['attack_success_rate']:.2f}%")

                c4, c5, c6 = st.columns(3)
                with c4:
                    st.metric("L2扰动", f"{metrics['perturbation_l2']:.4f}")
                with c5:
                    st.metric("Linf扰动", f"{metrics['perturbation_linf']:.4f}")
                with c6:
                    st.metric("SSIM", f"{metrics['ssim']:.4f}")

with tab2:
    st.header("结果分析")
    if "last_experiment" not in st.session_state:
        st.info("请先运行一次图像攻击实验。")
    else:
        exp = st.session_state["last_experiment"]
        metrics = exp["metrics"]
        st.subheader("图像分类攻击结果")
        st.info(f"模型：{exp['model']} | 数据集：{exp['dataset']} | 攻击：{exp['attack']} | 样本数：{exp['num_samples']}")
        if exp.get("label_space_mismatch", False):
            st.warning(
                f"类别空间不匹配（模型输出维度={exp.get('model_output_dim')}）。"
                " 请把指标理解为预测翻转强度，不要作为标准分类准确率。"
            )

        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("原始准确率", f"{metrics['original_accuracy']:.2f}%")
        with c2:
            st.metric("攻击后准确率", f"{metrics['adversarial_accuracy']:.2f}%")
        with c3:
            st.metric("预测翻转率", f"{metrics['attack_success_rate']:.2f}%")

        c4, c5, c6 = st.columns(3)
        with c4:
            st.metric("L2扰动", f"{metrics['perturbation_l2']:.4f}")
        with c5:
            st.metric("Linf扰动", f"{metrics['perturbation_linf']:.4f}")
        with c6:
            st.metric("置信度变化", f"{metrics['confidence_drop']:.4f}")

        for i in range(min(3, len(exp["original"]))):
            c1, c2, c3 = st.columns(3)
            with c1:
                img = exp["original"][i].detach().permute(1, 2, 0).numpy()
                st.image(np.clip(img, 0, 1), caption=f"原图 标签={exp['labels'][i].item()}", use_container_width=True)
            with c2:
                img = exp["adversarial"][i].detach().permute(1, 2, 0).numpy()
                st.image(np.clip(img, 0, 1), caption=f"对抗图 预测={exp['adv_preds'][i].item()}", use_container_width=True)
            with c3:
                pert = (exp["adversarial"][i] - exp["original"][i]).abs().detach().permute(1, 2, 0).numpy()
                st.image(to_perturbation_vis(pert), caption="扰动(归一化)", use_container_width=True)

with tab3:
    st.header("历史记录")
    dm = DatasetManager()
    experiments = dm.list_experiments()
    if not experiments:
        st.info("暂无保存实验。")
    else:
        def extract_metrics(meta: dict) -> dict:
            parsed = meta.get("metrics", {})
            if parsed:
                return parsed
            fallback_keys = [
                "original_accuracy",
                "adversarial_accuracy",
                "attack_success_rate",
                "prediction_flip_rate",
                "perturbation_l2",
                "perturbation_linf",
                "perturbation_mean",
                "ssim",
                "psnr",
                "confidence_drop",
                "original_confidence",
                "adversarial_confidence",
            ]
            return {k: meta[k] for k in fallback_keys if k in meta}

        records = []
        for exp_name in experiments:
            exp_path = os.path.join(dm.adversarial_dir, exp_name)
            try:
                meta = dm.load_adversarial_samples(exp_path)
            except Exception:
                continue
            ts_raw = str(meta.get("timestamp", ""))
            ts_dt = None
            ts_pretty = ts_raw
            try:
                ts_dt = datetime.strptime(ts_raw, "%Y%m%d_%H%M%S")
                ts_pretty = ts_dt.strftime("%Y-%m-%d %H:%M:%S")
            except ValueError:
                pass
            records.append(
                {
                    "name": exp_name,
                    "path": exp_path,
                    "metadata": meta,
                    "attack": str(meta.get("attack_name", "未知")),
                    "model": str(meta.get("model", "-")),
                    "dataset": str(meta.get("dataset", "-")),
                    "ts_dt": ts_dt,
                    "ts_raw": ts_raw,
                    "ts_pretty": ts_pretty,
                    "metrics": extract_metrics(meta),
                }
            )

        if not records:
            st.info("历史目录存在，但未找到有效的 metadata.json。")
        else:
            st.subheader("过滤")
            attack_options = sorted({r["attack"] for r in records})
            model_options = sorted({r["model"] for r in records})
            dataset_options = sorted({r["dataset"] for r in records})

            f1, f2, f3 = st.columns(3)
            with f1:
                selected_attacks = st.multiselect("攻击方法", attack_options, default=attack_options)
            with f2:
                selected_models = st.multiselect("模型", model_options, default=model_options)
            with f3:
                selected_datasets = st.multiselect("数据集", dataset_options, default=dataset_options)

            dated_records = [r for r in records if r["ts_dt"] is not None]
            date_start, date_end = None, None
            if dated_records:
                min_date = min(r["ts_dt"].date() for r in dated_records)
                max_date = max(r["ts_dt"].date() for r in dated_records)
                date_range = st.date_input("日期范围", value=(min_date, max_date))
                if isinstance(date_range, (tuple, list)) and len(date_range) == 2:
                    date_start, date_end = date_range

            filtered_records = []
            for r in records:
                if selected_attacks and r["attack"] not in selected_attacks:
                    continue
                if selected_models and r["model"] not in selected_models:
                    continue
                if selected_datasets and r["dataset"] not in selected_datasets:
                    continue
                if date_start is not None and date_end is not None and r["ts_dt"] is not None:
                    if not (date_start <= r["ts_dt"].date() <= date_end):
                        continue
                filtered_records.append(r)

            st.caption(f"筛选后实验数：{len(filtered_records)} / {len(records)}")

            if not filtered_records:
                st.warning("当前过滤条件下没有实验记录。")
            else:
                compare_mode = st.checkbox("开启实验对比模式（选择 2-3 条）", value=False)
                filtered_names = [r["name"] for r in filtered_records]

                if compare_mode:
                    default_pick = filtered_names[: min(2, len(filtered_names))]
                    compare_selected = st.multiselect("选择要对比的实验", filtered_names, default=default_pick)
                    if len(compare_selected) < 2:
                        st.info("请至少选择 2 条实验进行对比。")
                    elif len(compare_selected) > 3:
                        st.warning("最多选择 3 条实验进行并排对比。")
                    else:
                        cmp_records = [r for r in filtered_records if r["name"] in compare_selected]
                        cmp_rows = []
                        for r in cmp_records:
                            m = r["metrics"]
                            cmp_rows.append(
                                {
                                    "实验": r["name"],
                                    "时间": r["ts_pretty"],
                                    "攻击": r["attack"],
                                    "模型": r["model"],
                                    "数据集": r["dataset"],
                                    "样本数": r["metadata"].get("num_samples", "-"),
                                    "原始准确率(%)": m.get("original_accuracy"),
                                    "攻击后准确率(%)": m.get("adversarial_accuracy"),
                                    "预测翻转率(%)": m.get("attack_success_rate", m.get("prediction_flip_rate")),
                                    "L2扰动": m.get("perturbation_l2"),
                                    "Linf扰动": m.get("perturbation_linf"),
                                    "置信度变化": m.get("confidence_drop"),
                                }
                            )
                        st.subheader("实验对比表")
                        st.dataframe(pd.DataFrame(cmp_rows), use_container_width=True)

                selected = st.selectbox("选择实验（详细查看）", filtered_names)
                selected_record = next((r for r in filtered_records if r["name"] == selected), None)
                if selected_record is not None:
                    exp_path = selected_record["path"]
                    metadata = selected_record["metadata"]

                    st.subheader("实验摘要")
                    c1, c2, c3, c4 = st.columns(4)
                    with c1:
                        st.metric("攻击方法", selected_record["attack"])
                    with c2:
                        st.metric("样本数", str(metadata.get("num_samples", "-")))
                    with c3:
                        st.metric("模型", selected_record["model"])
                    with c4:
                        st.metric("数据集", selected_record["dataset"])

                    st.caption(f"时间：{selected_record['ts_pretty']}（原始时间戳：{selected_record['ts_raw']}）")
                    if "targeted" in metadata:
                        st.caption(f"目标攻击：{'是' if metadata.get('targeted') else '否'}")

                    parsed_metrics = selected_record["metrics"]
                    if parsed_metrics:
                        st.subheader("关键指标")
                        metric_cols = st.columns(3)
                        show_keys = [
                            ("original_accuracy", "原始准确率", "{:.2f}%"),
                            ("adversarial_accuracy", "攻击后准确率", "{:.2f}%"),
                            ("attack_success_rate", "预测翻转率", "{:.2f}%"),
                            ("perturbation_l2", "L2扰动", "{:.4f}"),
                            ("perturbation_linf", "Linf扰动", "{:.4f}"),
                            ("confidence_drop", "置信度变化", "{:.4f}"),
                        ]
                        for idx, (key, label, fmt) in enumerate(show_keys):
                            value = parsed_metrics.get(key)
                            if value is None and key == "attack_success_rate":
                                value = parsed_metrics.get("prediction_flip_rate")
                            if isinstance(value, (int, float)):
                                with metric_cols[idx % len(metric_cols)]:
                                    st.metric(label, fmt.format(value))

                        with st.expander("展开查看全部指标"):
                            st.json(parsed_metrics)

                    st.subheader("样本查看")
                    num_samples = int(metadata.get("num_samples", 0) or 0)
                    labels = metadata.get("labels", [])
                    preds = metadata.get("predictions", [])

                    if num_samples <= 0:
                        st.info("该实验没有可展示的样本。")
                    else:
                        if num_samples == 1:
                            sample_idx = 0
                            st.caption("样本索引：0（该实验仅有 1 个样本）")
                        else:
                            sample_idx = st.slider(
                                "样本索引",
                                0,
                                num_samples - 1,
                                0,
                                1,
                                key=f"history_sample_{selected}",
                            )
                        orig_path = os.path.join(exp_path, f"original_{sample_idx}.png")
                        adv_path = os.path.join(exp_path, f"adversarial_{sample_idx}.png")
                        pert_path = os.path.join(exp_path, f"perturbation_{sample_idx}.png")

                        cc1, cc2, cc3 = st.columns(3)
                        with cc1:
                            if os.path.exists(orig_path):
                                label_txt = labels[sample_idx] if sample_idx < len(labels) else "?"
                                st.image(orig_path, caption=f"原图 标签={label_txt}", use_container_width=True)
                            else:
                                st.warning("未找到原图文件。")
                        with cc2:
                            if os.path.exists(adv_path):
                                pred_txt = preds[sample_idx] if sample_idx < len(preds) else "?"
                                st.image(adv_path, caption=f"对抗图 预测={pred_txt}", use_container_width=True)
                            else:
                                st.warning("未找到对抗图文件。")
                        with cc3:
                            if os.path.exists(orig_path) and os.path.exists(adv_path):
                                try:
                                    orig_img_np = np.array(Image.open(orig_path).convert("RGB"), dtype=np.float32) / 255.0
                                    adv_img_np = np.array(Image.open(adv_path).convert("RGB"), dtype=np.float32) / 255.0
                                    diff_vis = to_perturbation_vis(np.abs(adv_img_np - orig_img_np))
                                    st.image(diff_vis, caption="扰动(归一化)", use_container_width=True)
                                except Exception:
                                    if os.path.exists(pert_path):
                                        st.image(pert_path, caption="扰动(归一化)", use_container_width=True)
                                    else:
                                        st.warning("未找到扰动图文件。")
                            elif os.path.exists(pert_path):
                                st.image(pert_path, caption="扰动(归一化)", use_container_width=True)
                            else:
                                st.warning("未找到扰动图文件。")

                    with st.expander("查看原始 metadata.json"):
                        st.json(metadata)

with tab4:
    st.header("使用说明")
    st.markdown(
        """
        - 本页面聚焦图片模态攻击：FGSM / PGD / C&W。
        - 自定义图片默认使用 ImageNet 预训练模型，可直接上传图片演示。
        - CIFAR-10 / MNIST 批量实验需要先运行 `scripts/train_classifier.py` 训练微调权重。
        - 没有微调权重时，平台会停止标准数据集实验，避免把随机分类头结果当成有效攻击结果。
        - 将 epsilon 设为 0 可以验证模型本身：原始准确率应与攻击后准确率一致，预测翻转率和扰动应为 0。
        """
    )

st.markdown("---")
st.caption("图像对抗攻击平台 v1.3")
