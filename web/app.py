"""
基于图像模态的多模态模型攻击与安全评估平台
"""
from __future__ import annotations

import inspect
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
from PIL import Image
import streamlit as st
import torch
import torch.nn.functional as F
from torchvision import transforms

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.attacks import FGSM, PGD, CarliniWagner, CLIPMultimodalAttack, BLIPCaptionAttack
from src.data_manager import DatasetManager
from src.evaluation import (
    AttackEvaluator,
    EvaluationAdapterRegistry,
    load_evaluation_adapter,
    CLIPMultimodalEvaluator,
    clip_attack_decision,
    blip_attack_decision,
)
from src.models import ModelLoader, load_model


def to_perturbation_vis(perturbation: np.ndarray) -> np.ndarray:
    max_val = float(perturbation.max())
    if max_val <= 1e-12:
        return perturbation
    return np.clip(perturbation / max_val, 0.0, 1.0)


def tensor_to_image_np(image: torch.Tensor) -> np.ndarray:
    return np.clip(image.detach().cpu().permute(1, 2, 0).numpy(), 0.0, 1.0)


def tensors_to_perturbation_np(original: torch.Tensor, adversarial: torch.Tensor) -> np.ndarray:
    original_cpu = align_tensor_image(original, adversarial).detach().cpu()
    adversarial_cpu = adversarial.detach().cpu()
    perturbation = (adversarial_cpu - original_cpu).abs().permute(1, 2, 0).numpy()
    return to_perturbation_vis(perturbation)


def align_tensor_image(original: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    if original.dim() != reference.dim():
        raise ValueError(f"Tensor ranks do not match: {original.dim()} vs {reference.dim()}")
    if original.shape[-2:] == reference.shape[-2:]:
        return original
    if original.dim() == 3:
        resized = F.interpolate(
            original.unsqueeze(0),
            size=reference.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        return resized.squeeze(0)
    if original.dim() == 4:
        return F.interpolate(
            original,
            size=reference.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
    raise ValueError(f"Unsupported tensor shape for alignment: {tuple(original.shape)}")


def infer_saved_sample_count(experiment_path: str) -> int:
    count = 0
    if not os.path.isdir(experiment_path):
        return 0
    for file_name in os.listdir(experiment_path):
        if file_name.startswith("original_") and file_name.endswith(".png"):
            count += 1
    return count


def instantiate_with_supported_kwargs(cls, /, *args, **kwargs):
    """Instantiate a class while ignoring unsupported keyword arguments."""
    signature = inspect.signature(cls.__init__)
    supported = {
        key: value
        for key, value in kwargs.items()
        if key in signature.parameters
    }
    return cls(*args, **supported)


def sync_uploaded_collection_annotations(data_manager: DatasetManager, collection_dir: str) -> None:
    manifest = data_manager.load_uploaded_collection(collection_dir)
    annotations = {}
    collection_name = os.path.basename(collection_dir)
    for sample in manifest.get("samples", []):
        annotations[f"{collection_name}/{sample.get('file_name')}"] = {
            "label": sample.get("label"),
            "split": sample.get("split", "unspecified"),
            "notes": sample.get("notes", ""),
            "tags": sample.get("tags", []),
        }
    data_manager.update_annotations(annotations, annotation_file="sample_annotations.json", merge=True)


def sanitize_metrics(metrics: dict) -> dict:
    cleaned = {}
    for key, value in metrics.items():
        if isinstance(value, torch.Tensor):
            cleaned[key] = value.detach().cpu().tolist() if value.numel() > 1 else float(value.item())
        elif isinstance(value, np.ndarray):
            cleaned[key] = value.tolist()
        elif isinstance(value, (np.floating, float)):
            cleaned[key] = float(value)
        elif isinstance(value, (np.integer, int)):
            cleaned[key] = int(value)
        else:
            cleaned[key] = value
    return cleaned


def make_metric_dataframe(target_family: str, metrics: dict) -> pd.DataFrame:
    decision = metrics.get("decision", {})
    if target_family == "CLIP 图文对齐":
        rows = [
            {"指标": "源相似度下降", "值": float(metrics.get("similarity_drop", 0.0)), "类别": "相似度"},
            {"指标": "目标相似度提升", "值": float(metrics.get("decision", {}).get("target_similarity_gain", 0.0)), "类别": "相似度"},
            {"指标": "L2 扰动", "值": float(metrics.get("perturbation_l2", 0.0)), "类别": "扰动"},
            {"指标": "Linf 扰动", "值": float(metrics.get("perturbation_linf", 0.0)), "类别": "扰动"},
        ]
    else:
        rows = [
            {"指标": "描述变化率", "值": float(metrics.get("caption_change_rate", 0.0)), "类别": "描述"},
            {"指标": "目标重叠提升", "值": float(decision.get("target_overlap_gain", 0.0)), "类别": "描述"},
            {"指标": "源重叠下降", "值": float(decision.get("source_overlap_drop", 0.0)), "类别": "描述"},
            {"指标": "L2 扰动", "值": float(metrics.get("perturbation_l2", 0.0)), "类别": "扰动"},
            {"指标": "Linf 扰动", "值": float(metrics.get("perturbation_linf", 0.0)), "类别": "扰动"},
        ]
    return pd.DataFrame(rows)


def render_metric_charts(target_family: str, metrics: dict, chart_key_prefix: str) -> None:
    metric_df = make_metric_dataframe(target_family, metrics)
    st.dataframe(metric_df, use_container_width=True, hide_index=True)

    fig = px.bar(
        metric_df,
        x="指标",
        y="值",
        color="类别",
        title="关键指标总览",
        template="plotly_dark",
    )
    fig.update_layout(margin=dict(l=20, r=20, t=50, b=20), legend_title_text="")
    st.plotly_chart(fig, use_container_width=True, key=f"{chart_key_prefix}_bar")

    loss_history = metrics.get("loss_history", [])
    if isinstance(loss_history, list) and loss_history:
        loss_df = pd.DataFrame({"迭代步": list(range(1, len(loss_history) + 1)), "损失": loss_history})
        loss_fig = px.line(
            loss_df,
            x="迭代步",
            y="损失",
            markers=True,
            title="攻击优化过程",
            template="plotly_dark",
        )
        loss_fig.update_layout(margin=dict(l=20, r=20, t=50, b=20))
        st.plotly_chart(loss_fig, use_container_width=True, key=f"{chart_key_prefix}_loss")


def dataset_to_key(dataset_name: str) -> str:
    return dataset_name.lower().replace("-", "")


def dataset_to_num_classes(dataset_name: str) -> int:
    if dataset_name in {"CIFAR-10", "MNIST"}:
        return 10
    return 1000


def classification_model_choices() -> list[str]:
    return ModelLoader.get_supported_models(include_clip=False, include_caption=False)


def clip_model_choices() -> list[str]:
    return [
        name
        for name in ModelLoader.get_supported_models(include_clip=True, include_caption=False)
        if "clip" in name.lower()
    ]


def caption_model_choices() -> list[str]:
    return [
        name
        for name in ModelLoader.get_supported_models(include_clip=False, include_caption=True)
        if "blip" in name.lower()
    ]


def multimodal_model_choices() -> list[str]:
    return clip_model_choices() + caption_model_choices()


def preprocess_custom_image(image: Image.Image) -> torch.Tensor:
    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ]
    )
    return transform(image).unsqueeze(0)


@st.cache_resource(show_spinner=False)
def load_cached_classification_model(
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


@st.cache_resource(show_spinner=False)
def load_cached_evaluation_adapter(
    adapter_key: str,
    model_name: str,
    device: str,
):
    return load_evaluation_adapter(
        adapter_key=adapter_key,
        model_name=model_name,
        device=device,
    )


@st.cache_resource(show_spinner=False)
def load_cached_clip_bundle(model_name: str, device: str):
    return ModelLoader.load_model(model_name, pretrained=True, device=device)


@st.cache_resource(show_spinner=False)
def load_cached_blip_bundle(model_name: str, device: str):
    return ModelLoader.load_caption_model(model_name, device=device)


def resolve_classification_model(model_name: str, dataset_name: str, device: str):
    if dataset_name == "自定义图片":
        model = load_cached_classification_model(
            model_name=model_name,
            device=device,
            num_classes=1000,
            dataset_name="ImageNet",
            checkpoint_version="imagenet",
        )
        return model, None, "loaded"

    num_classes = dataset_to_num_classes(dataset_name)
    checkpoint_path = ModelLoader.get_finetune_checkpoint_path(model_name, dataset_name)
    checkpoint_version = (
        str(os.path.getmtime(checkpoint_path))
        if checkpoint_path and os.path.exists(checkpoint_path)
        else "missing"
    )
    model = load_cached_classification_model(
        model_name=model_name,
        device=device,
        num_classes=num_classes,
        dataset_name=dataset_name,
        checkpoint_version=checkpoint_version,
    )
    return model, checkpoint_path, str(getattr(model, "fine_tuned_checkpoint_status", "unknown"))


def build_attacker(model, attack_method: str, attack_params: dict, device: str):
    if attack_method == "FGSM":
        return FGSM(model, device=device, **attack_params)
    if attack_method == "PGD":
        return PGD(model, device=device, **attack_params)
    return CarliniWagner(model, device=device, **attack_params)


def show_metrics(metrics: dict) -> None:
    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("原始准确率", f"{metrics['original_accuracy']:.2f}%")
    with c2:
        st.metric("攻击后准确率", f"{metrics['adversarial_accuracy']:.2f}%")
    with c3:
        st.metric("攻击成功率", f"{metrics['attack_success_rate']:.2f}%")

    c4, c5, c6 = st.columns(3)
    with c4:
        st.metric("L2 扰动", f"{metrics['perturbation_l2']:.4f}")
    with c5:
        st.metric("Linf 扰动", f"{metrics['perturbation_linf']:.4f}")
    with c6:
        st.metric("SSIM", f"{metrics['ssim']:.4f}")


def store_last_experiment(
    model_name: str,
    dataset_name: str,
    attack_method: str,
    metrics: dict,
    original: torch.Tensor,
    adversarial: torch.Tensor,
    labels: torch.Tensor,
    orig_preds: torch.Tensor,
    adv_preds: torch.Tensor,
    save_dir: str | None = None,
    output_dim: int | None = None,
) -> None:
    expected_classes = {"CIFAR-10": 10, "MNIST": 10}.get(dataset_name)
    label_space_mismatch = (
        expected_classes is not None and output_dim is not None and output_dim != expected_classes
    )
    st.session_state["last_experiment"] = {
        "model": model_name,
        "dataset": dataset_name,
        "attack": attack_method,
        "metrics": metrics,
        "num_samples": int(original.shape[0]),
        "original": original[:10].detach().cpu(),
        "adversarial": adversarial[:10].detach().cpu(),
        "labels": labels[:10].detach().cpu(),
        "orig_preds": orig_preds[:10].detach().cpu(),
        "adv_preds": adv_preds[:10].detach().cpu(),
        "save_dir": save_dir,
        "multimodal_evaluations": {},
        "label_space_mismatch": label_space_mismatch,
        "model_output_dim": output_dim,
    }


def persist_multimodal_evaluation(experiment: dict, key: str, payload: dict) -> None:
    experiment["multimodal_evaluations"][key] = sanitize_metrics(payload)

    save_dir = experiment.get("save_dir")
    if not save_dir:
        return

    data_manager = DatasetManager()
    try:
        data_manager.update_experiment_metadata(
            save_dir,
            {"multimodal_evaluations": {key: sanitize_metrics(payload)}},
            merge=True,
        )
    except Exception:
        pass


def store_last_multimodal_experiment(
    target_family: str,
    model_name: str,
    attack_method: str,
    attack_mode: str,
    metrics: dict,
    original: torch.Tensor,
    adversarial: torch.Tensor,
    source_text: str,
    target_text: str | None,
    targeted: bool,
    save_dir: str | None = None,
    extra: dict | None = None,
    multimodal_evaluations: dict | None = None,
) -> None:
    original = align_tensor_image(original, adversarial)
    st.session_state["last_experiment"] = {
        "task_type": "multimodal",
        "target_family": target_family,
        "model": model_name,
        "dataset": "自定义图片",
        "attack": attack_method,
        "attack_mode": attack_mode,
        "metrics": metrics,
        "num_samples": 1,
        "original": original[:1].detach().cpu(),
        "adversarial": adversarial[:1].detach().cpu(),
        "source_text": source_text,
        "target_text": target_text,
        "targeted": targeted,
        "save_dir": save_dir,
        "multimodal_evaluations": multimodal_evaluations or {},
        "extra": extra or {},
    }


def _save_multimodal_record(
    data_manager: DatasetManager,
    experiment_name: str,
    original: torch.Tensor,
    adversarial: torch.Tensor,
    metadata: dict,
) -> str:
    original = align_tensor_image(original, adversarial)
    return data_manager.save_experiment_record(
        experiment_name,
        metadata=metadata,
        original_image=original[0].detach().cpu(),
        adversarial_image=adversarial[0].detach().cpu(),
    )


def render_multimodal_attack_tab(
    target_family: str,
    model_name: str,
    attack_method: str,
    attack_mode: str,
    attack_params: dict,
    targeted: bool,
    device: str,
) -> None:
    st.header("多模态攻击实验")
    st.caption("主线改为 CLIP / BLIP 多模态目标，ResNet 不再出现在默认入口。")

    col1, col2 = st.columns([1, 1], gap="large")
    data_manager = DatasetManager()

    with col1:
        st.subheader("输入")
        uploaded_file = st.file_uploader("上传图片", type=["png", "jpg", "jpeg"], key="multimodal_upload")
        image = None
        image_tensor = None
        if uploaded_file is not None:
            image = Image.open(uploaded_file).convert("RGB")
            image_tensor = preprocess_custom_image(image)
            st.image(image, caption="上传图片", use_container_width=True)

        source_text = st.text_input(
            "源文本提示",
            value="a photo of a cat" if target_family == "CLIP 图文对齐" else "describe the image in detail",
            key="source_prompt_input",
        )
        target_text = st.text_input(
            "目标文本提示",
            value="a photo of a dog" if target_family == "CLIP 图文对齐" else "a detailed caption describing a different scene",
            key="target_prompt_input",
        )

        st.info(
            "CLIP 会优化图文相似度；BLIP 会优化图像描述生成。"
        )
        if target_family == "BLIP 图像描述" and attack_method == "FGSM":
            st.warning("BLIP 目标下 FGSM 仅作为单步基线，通常不容易达到统一阈值；建议切换 PGD。")
        if target_family == "BLIP 图像描述":
            st.caption("建议目标文本写成完整 caption，例如 'a dog lying on a couch'，而不是只写单个词。")

    with col2:
        st.subheader("执行")
        if st.button("开始多模态攻击", type="primary", use_container_width=True):
            if uploaded_file is None or image is None or image_tensor is None:
                st.error("请先上传一张图片。")
                st.stop()

            source_clean = source_text.strip()
            target_clean = target_text.strip()
            if targeted and (not target_clean or source_clean == target_clean):
                st.error("目标攻击时，源文本和目标文本必须不同。请修改目标文本后再运行。")
                st.stop()

            with st.spinner("加载模型并执行多模态攻击..."):
                if target_family == "CLIP 图文对齐":
                    clip_bundle = load_cached_clip_bundle(model_name, device)
                    attacker = CLIPMultimodalAttack(
                        clip_bundle.clip_model,
                        clip_bundle.processor,
                        device=device,
                        image_epsilon=attack_params["epsilon"],
                        image_alpha=attack_params.get("alpha", attack_params["epsilon"]),
                        num_steps=attack_params.get("num_steps", 1),
                    )
                    adv_image, adv_text_feat, attack_info = attacker.generate(
                        image_tensor,
                        source_text=source_clean or "a photo of a cat",
                        target_text=target_clean or None,
                        mode=attack_mode,
                        targeted=targeted,
                    )
                    evaluator = CLIPMultimodalEvaluator(clip_bundle.clip_model, clip_bundle.processor, device=device)
                    eval_metrics = evaluator.evaluate(
                        image_tensor,
                        adv_image,
                        source_text=source_clean or "a photo of a cat",
                        target_text=target_clean or None,
                        adversarial_text_feature=adv_text_feat,
                        targeted=targeted,
                    )
                    decision = clip_attack_decision(eval_metrics, targeted=targeted)
                    metrics = {
                        **sanitize_metrics(attack_info),
                        **sanitize_metrics(eval_metrics),
                        "decision": sanitize_metrics(decision),
                    }
                    save_dir = _save_multimodal_record(
                        data_manager,
                        experiment_name=f"clip_{attack_method.lower()}_{attack_mode}",
                        original=image_tensor,
                        adversarial=adv_image,
                        metadata={
                            "task_type": "multimodal",
                            "target_family": target_family,
                            "dataset": "自定义图片",
                            "model": model_name,
                            "attack": attack_method,
                            "attack_mode": attack_mode,
                            "source_text": source_text,
                            "target_text": target_text,
                            "targeted": targeted,
                            "metrics": metrics,
                        },
                    )
                    persist_multimodal_evaluation(
                        {"multimodal_evaluations": {}, "save_dir": save_dir},
                        "clip",
                        metrics,
                    )
                    store_last_multimodal_experiment(
                        target_family=target_family,
                        model_name=model_name,
                        attack_method=attack_method,
                        attack_mode=attack_mode,
                        metrics=metrics,
                        original=image_tensor,
                        adversarial=adv_image,
                        source_text=source_text,
                        target_text=target_clean if targeted else None,
                        targeted=targeted,
                        save_dir=save_dir,
                        extra={"adv_text_feat": adv_text_feat.detach().cpu()},
                        multimodal_evaluations={"clip": metrics},
                    )
                    st.success("CLIP 多模态攻击完成。")
                    if decision["attack_success"]:
                        st.success(f"统一判定：攻击成功。{decision['reason']}")
                    else:
                        st.warning(f"统一判定：攻击未达阈值。{decision['reason']}")
                    p1, p2, p3 = st.columns(3)
                    with p1:
                        st.image(image, caption="原图", use_container_width=True)
                    with p2:
                        st.image(tensor_to_image_np(adv_image[0]), caption="对抗图", use_container_width=True)
                    with p3:
                        st.image(tensors_to_perturbation_np(image_tensor[0], adv_image[0]), caption="扰动图", use_container_width=True)
                    st.metric("源相似度", f"{metrics.get('orig_source_similarity', 0.0):.4f}")
                    st.metric("对抗相似度", f"{metrics.get('adv_source_similarity', 0.0):.4f}")
                    if targeted and "adv_target_similarity" in metrics:
                        st.metric("目标相似度", f"{metrics.get('adv_target_similarity', 0.0):.4f}")
                    render_metric_charts(target_family, metrics, "clip_attack")
                    with st.expander("完整指标"):
                        st.json(sanitize_metrics(metrics))
                    st.info(f"实验已保存到：{save_dir}")
                else:
                    blip_bundle = load_cached_blip_bundle(model_name, device)
                    attacker = instantiate_with_supported_kwargs(
                        BLIPCaptionAttack,
                        blip_bundle,
                        blip_bundle.processor,
                        device=device,
                        epsilon=attack_params["epsilon"],
                        alpha=attack_params.get("alpha", attack_params["epsilon"]),
                        num_steps=attack_params.get("num_steps", 1),
                        num_restarts=attack_params.get("num_restarts", 3),
                        source_weight=attack_params.get("source_weight", 0.35),
                    )
                    adv_image, attack_info = attacker.generate(
                        image_tensor,
                        source_text=source_clean,
                        target_text=target_clean or None,
                        targeted=targeted,
                    )
                    blip_adapter = load_cached_evaluation_adapter("blip", model_name, device)
                    eval_metrics = blip_adapter.evaluate(
                        image_tensor,
                        adv_image,
                        prompt=source_clean or None,
                    )
                    decision = blip_attack_decision(
                        original_caption=str(attack_info.get("original_caption", "")),
                        adversarial_caption=str(attack_info.get("adversarial_caption", "")),
                        source_text=source_clean,
                        target_text=target_clean if targeted else None,
                        targeted=targeted,
                    )
                    metrics = {
                        **sanitize_metrics(attack_info),
                        **sanitize_metrics(eval_metrics),
                        "decision": sanitize_metrics(decision),
                    }
                    save_dir = _save_multimodal_record(
                        data_manager,
                        experiment_name=f"blip_{attack_method.lower()}",
                        original=image_tensor,
                        adversarial=adv_image,
                        metadata={
                            "task_type": "multimodal",
                            "target_family": target_family,
                            "dataset": "自定义图片",
                            "model": model_name,
                            "attack": attack_method,
                        "source_text": source_text,
                            "target_text": target_clean,
                            "targeted": targeted,
                            "metrics": metrics,
                        },
                    )
                    persist_multimodal_evaluation(
                        {"multimodal_evaluations": {}, "save_dir": save_dir},
                        "blip",
                        metrics,
                    )
                    store_last_multimodal_experiment(
                        target_family=target_family,
                        model_name=model_name,
                        attack_method=attack_method,
                        attack_mode="image",
                        metrics=metrics,
                        original=image_tensor,
                        adversarial=adv_image,
                        source_text=source_text,
                        target_text=target_clean if targeted else None,
                        targeted=targeted,
                        save_dir=save_dir,
                        extra={"original_caption": attack_info.get("original_caption"), "adversarial_caption": attack_info.get("adversarial_caption")},
                        multimodal_evaluations={"blip": metrics},
                    )
                    st.success("BLIP 多模态攻击完成。")
                    if decision["attack_success"]:
                        st.success(f"统一判定：攻击成功。{decision['reason']}")
                    else:
                        st.warning(f"统一判定：攻击未达阈值。{decision['reason']}")
                    p1, p2, p3 = st.columns(3)
                    with p1:
                        st.image(image, caption="原图", use_container_width=True)
                    with p2:
                        st.image(tensor_to_image_np(adv_image[0]), caption="对抗图", use_container_width=True)
                    with p3:
                        st.image(tensors_to_perturbation_np(image_tensor[0], adv_image[0]), caption="扰动图", use_container_width=True)
                    st.write(f"原图描述：{attack_info.get('original_caption', '')}")
                    st.write(f"对抗图描述：{attack_info.get('adversarial_caption', '')}")
                    render_metric_charts(target_family, metrics, "blip_attack")
                    with st.expander("完整指标"):
                        st.json(sanitize_metrics(metrics))
                    st.info(f"实验已保存到：{save_dir}")


def render_multimodal_results_tab(
    device: str,
) -> None:
    st.header("结果分析")
    exp = st.session_state.get("last_experiment")
    if exp is None:
        st.info("请先在“多模态攻击实验”页运行一次攻击。")
        return

    if exp.get("task_type") != "multimodal":
        st.warning("当前会话里保存的是旧的分类基线实验记录。请切换到新的多模态实验并重新运行。")
        return

    metrics = exp.get("metrics", {})
    st.info(
        f"目标：{exp.get('target_family', '-') } | 模型：{exp.get('model', '-') } | "
        f"攻击：{exp.get('attack', '-') } | 方式：{exp.get('attack_mode', '-') }"
    )
    if exp.get("source_text"):
        st.write(f"源文本：{exp.get('source_text')}")
    if exp.get("targeted") and exp.get("target_text"):
        st.write(f"目标文本：{exp.get('target_text')}")

    decision = metrics.get("decision", {})
    if not decision:
        if exp.get("target_family") == "CLIP 图文对齐":
            decision = clip_attack_decision(metrics, targeted=bool(exp.get("targeted", False)))
        else:
            decision = blip_attack_decision(
                original_caption=str(metrics.get("original_caption", "")),
                adversarial_caption=str(metrics.get("adversarial_caption", "")),
                source_text=exp.get("source_text", ""),
                target_text=exp.get("target_text"),
                targeted=bool(exp.get("targeted", False)),
            )
        metrics["decision"] = sanitize_metrics(decision)

    if decision.get("attack_success"):
        st.success(f"统一判定：攻击成功。{decision.get('reason', '')}")
    else:
        st.warning(f"统一判定：攻击未达阈值。{decision.get('reason', '')}")

    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("扰动 L2", f"{float(metrics.get('perturbation_l2', 0.0)):.4f}")
    with c2:
        st.metric("扰动 Linf", f"{float(metrics.get('perturbation_linf', 0.0)):.4f}")
    with c3:
        if exp.get("target_family") == "CLIP 图文对齐":
            st.metric("源相似度下降", f"{float(metrics.get('similarity_drop', 0.0)):.4f}")
        else:
            st.metric("描述变化率", f"{float(metrics.get('caption_change_rate', 0.0)):.2f}%")

    render_metric_charts(exp.get("target_family", "CLIP 图文对齐"), metrics, "results")

    st.subheader("攻击前后对比")
    d1, d2, d3 = st.columns(3)
    with d1:
        st.image(tensor_to_image_np(exp["original"][0]), caption="原图", use_container_width=True)
    with d2:
        st.image(tensor_to_image_np(exp["adversarial"][0]), caption="对抗图", use_container_width=True)
    with d3:
        st.image(tensors_to_perturbation_np(exp["original"][0], exp["adversarial"][0]), caption="扰动图", use_container_width=True)

    multimodal_evaluations = exp.get("multimodal_evaluations", {})
    if multimodal_evaluations:
        with st.expander("查看当前实验已记录的多模态评测", expanded=False):
            st.json(sanitize_metrics(multimodal_evaluations))

    if exp.get("target_family") == "BLIP 图像描述":
        orig_captions = metrics.get("original_captions", [])
        adv_captions = metrics.get("adversarial_captions", [])
        if orig_captions and adv_captions:
            st.subheader("BLIP 描述对比")
            rows = []
            for idx, (orig_caption, adv_caption) in enumerate(zip(orig_captions, adv_captions)):
                rows.append(
                    {
                        "样本": idx,
                        "原图描述": orig_caption,
                        "对抗图描述": adv_caption,
                    }
                )
            st.dataframe(pd.DataFrame(rows), use_container_width=True)

    if exp.get("target_family") == "CLIP 图文对齐":
        clip_decision = decision
        d1, d2 = st.columns(2)
        with d1:
            st.metric("源相似度下降", f"{float(clip_decision.get('source_similarity_drop', 0.0)):.4f}")
        with d2:
            st.metric("目标相似度提升", f"{float(clip_decision.get('target_similarity_gain', 0.0)):.4f}")
    else:
        blip_decision = decision
        d1, d2, d3 = st.columns(3)
        with d1:
            st.metric("目标重叠提升", f"{float(blip_decision.get('target_overlap_gain', 0.0)):.4f}")
        with d2:
            st.metric("源重叠下降", f"{float(blip_decision.get('source_overlap_drop', 0.0)):.4f}")
        with d3:
            st.metric("关键词命中", "是" if blip_decision.get("target_keyword_hit") else "否")
        st.caption("BLIP 目标攻击以‘目标关键词命中 + 目标重叠提升’为主判据；仅描述变化不代表真正命中目标。")

    st.subheader("交叉评测")
    clip_options = clip_model_choices()
    caption_options = caption_model_choices()

    if clip_options:
        selected_clip_model = st.selectbox("CLIP 模型", clip_options, key="results_clip_model_select")
        clip_prompt = st.text_input(
            "CLIP 提示词",
            value=exp.get("source_text", "a photo of the main object"),
            key="results_clip_prompt",
        )
        if st.button("计算 CLIP 验证", use_container_width=True, key="results_clip_eval_btn"):
            with st.spinner("加载 CLIP 并计算相似度..."):
                clip_bundle = load_cached_clip_bundle(selected_clip_model, device)
                clip_eval = CLIPMultimodalEvaluator(clip_bundle.clip_model, clip_bundle.processor, device=device)
                clip_metrics = clip_eval.evaluate(
                    exp["original"],
                    exp["adversarial"],
                    source_text=clip_prompt,
                    target_text=exp.get("target_text"),
                    targeted=bool(exp.get("targeted", False)),
                )
            persist_multimodal_evaluation(exp, "clip", clip_metrics)
            st.json(sanitize_metrics(clip_metrics))

    if caption_options:
        selected_caption_model = st.selectbox("BLIP 模型", caption_options, key="results_blip_model_select")
        caption_prompt = st.text_input(
            "BLIP 提示词",
            value=exp.get("source_text", ""),
            key="results_blip_prompt",
        )
        if st.button("计算 BLIP 验证", use_container_width=True, key="results_blip_eval_btn"):
            with st.spinner("加载 BLIP 并生成描述..."):
                blip_adapter = load_cached_evaluation_adapter("blip", selected_caption_model, device)
                blip_metrics = blip_adapter.evaluate(
                    exp["original"],
                    exp["adversarial"],
                    prompt=caption_prompt.strip() or None,
                )
            persist_multimodal_evaluation(exp, "blip", blip_metrics)
            rows = []
            for idx, (orig_caption, adv_caption) in enumerate(
                zip(blip_metrics.get("original_captions", []), blip_metrics.get("adversarial_captions", []))
            ):
                rows.append(
                    {
                        "样本": idx,
                        "原图描述": orig_caption,
                        "对抗图描述": adv_caption,
                    }
                )
            if rows:
                st.dataframe(pd.DataFrame(rows), use_container_width=True)
            st.json(sanitize_metrics(blip_metrics))


def render_sample_management_tab() -> None:
    st.header("样本管理")
    dm = DatasetManager()

    st.subheader("上传与标注")
    collection_name = st.text_input(
        "集合名称",
        value=f"multimodal_samples_{datetime.now().strftime('%Y%m%d')}",
        key="sample_collection_name",
    )
    uploaded_files = st.file_uploader(
        "上传图像样本",
        type=["png", "jpg", "jpeg"],
        accept_multiple_files=True,
        key="sample_mgmt_uploads",
    )

    if uploaded_files:
        upload_rows = []
        for file_obj in uploaded_files:
            upload_rows.append(
                {
                    "file_name": getattr(file_obj, "name", "sample.png"),
                    "label": "",
                    "split": "train",
                    "notes": "",
                    "tags": "",
                }
            )
        upload_df = st.data_editor(pd.DataFrame(upload_rows), use_container_width=True, num_rows="fixed")
        if st.button("保存上传集合", use_container_width=True, key="save_sample_collection_btn"):
            save_dir = dm.save_uploaded_samples(
                uploaded_files,
                annotations=upload_df.to_dict("records"),
                collection_name=collection_name.strip() or None,
            )
            sync_uploaded_collection_annotations(dm, save_dir)
            st.success(f"样本集合已保存：{save_dir}")
            st.session_state["selected_sample_collection"] = os.path.basename(save_dir)

    st.subheader("已有集合")
    collections = dm.list_uploaded_collections()
    if not collections:
        st.info("暂无上传样本集合。")
        return

    default_collection = st.session_state.get("selected_sample_collection", collections[0])
    selected_collection = st.selectbox(
        "选择集合",
        collections,
        index=collections.index(default_collection) if default_collection in collections else 0,
        key="sample_collection_select",
    )
    collection_dir = os.path.join(dm.raw_dir, "uploads", selected_collection)
    manifest = dm.load_uploaded_collection(collection_dir)
    samples_df = pd.DataFrame(manifest.get("samples", []))
    if samples_df.empty:
        st.info("该集合没有可显示的样本。")
        return

    filter_col1, filter_col2 = st.columns(2)
    with filter_col1:
        label_options = sorted(
            {
                str(item)
                for item in samples_df.get("label", pd.Series(dtype=str)).dropna().tolist()
                if str(item).strip()
            }
        )
        selected_labels = st.multiselect(
            "按标签筛选",
            label_options,
            default=label_options,
            key="sample_label_filter",
        )
    with filter_col2:
        keyword = st.text_input("按文件名/备注筛选", value="", key="sample_keyword_filter")

    filtered_df = samples_df.copy()
    if selected_labels:
        filtered_df = filtered_df[filtered_df["label"].astype(str).isin(selected_labels)]
    if keyword.strip():
        kw = keyword.strip().lower()
        filtered_df = filtered_df[
            filtered_df["file_name"].astype(str).str.lower().str.contains(kw, na=False)
            | filtered_df["notes"].astype(str).str.lower().str.contains(kw, na=False)
        ]

    if filtered_df.empty:
        st.warning("当前筛选条件下没有样本。")
        return

    editable_df = filtered_df[["file_name", "label", "split", "notes", "tags"]].copy()
    editable_df["tags"] = editable_df["tags"].apply(
        lambda value: ", ".join(value) if isinstance(value, list) else str(value or "")
    )
    edited_df = st.data_editor(editable_df, use_container_width=True, num_rows="fixed", key="sample_manifest_editor")

    save_edit_col, version_col, export_col = st.columns(3)
    with save_edit_col:
        if st.button("保存标注修改", use_container_width=True, key="save_sample_annotations_btn"):
            dm.update_uploaded_collection_manifest(collection_dir, edited_df.to_dict("records"))
            sync_uploaded_collection_annotations(dm, collection_dir)
            st.success("标注已更新。")

    with version_col:
        version_name = st.text_input(
            "版本名",
            value=f"v_{selected_collection}",
            key="sample_version_name",
        )
        if st.button("创建版本快照", use_container_width=True, key="create_sample_version_btn"):
            version_dir = dm.create_dataset_version(
                collection_dir,
                version_name=version_name.strip() or None,
                metadata={"source_collection": selected_collection, "num_samples": int(len(samples_df))},
            )
            st.success(f"已创建版本：{version_dir}")

    with export_col:
        if st.button("导出清单", use_container_width=True, key="export_manifest_btn"):
            st.session_state["sample_manifest_export"] = selected_collection

    if st.session_state.get("sample_manifest_export") == selected_collection:
        csv_bytes = dm.export_uploaded_collection_manifest(collection_dir, export_format="csv")
        json_bytes = dm.export_uploaded_collection_manifest(collection_dir, export_format="json")
        st.download_button(
            "下载 CSV 清单",
            data=csv_bytes,
            file_name=f"{selected_collection}_manifest.csv",
            mime="text/csv",
            use_container_width=True,
        )
        st.download_button(
            "下载 JSON 清单",
            data=json_bytes,
            file_name=f"{selected_collection}_manifest.json",
            mime="application/json",
            use_container_width=True,
        )

    st.subheader("样本预览")
    preview_names = filtered_df["file_name"].tolist()
    selected_file = st.selectbox("选择样本", preview_names, key="sample_preview_select")
    preview_row = filtered_df[filtered_df["file_name"] == selected_file].iloc[0]
    preview_path = os.path.join(collection_dir, preview_row["file_name"])
    c1, c2 = st.columns([1, 1], gap="large")
    with c1:
        if os.path.exists(preview_path):
            st.image(preview_path, caption=preview_row["file_name"], use_container_width=True)
        else:
            st.warning("未找到预览图像。")
    with c2:
        st.json(preview_row.to_dict())

    st.subheader("实验索引导出")
    experiment_rows = dm.build_experiment_index()
    if experiment_rows:
        exp_df = pd.DataFrame(experiment_rows)
        st.dataframe(exp_df, use_container_width=True)
        exp_csv = dm.export_experiment_index("csv")
        exp_json = dm.export_experiment_index("json")
        st.download_button(
            "导出实验索引 CSV",
            data=exp_csv,
            file_name="experiment_index.csv",
            mime="text/csv",
            use_container_width=True,
        )
        st.download_button(
            "导出实验索引 JSON",
            data=exp_json,
            file_name="experiment_index.json",
            mime="application/json",
            use_container_width=True,
        )


def render_attack_tab(
    attack_method: str,
    model_name: str,
    dataset_name: str,
    attack_params: dict,
    targeted: bool,
    device: str,
) -> None:
    st.header("攻击实验")
    st.caption("主线保留图像分类攻击，结果页提供黑盒迁移评测与多模态验证。")

    col1, col2 = st.columns([1, 1], gap="large")
    data_manager = DatasetManager()

    with col1:
        st.subheader("输入")
        uploaded_file = None
        image = None
        image_tensor = None
        true_label = 0
        target_label = None
        num_samples = 10

        if dataset_name == "自定义图片":
            uploaded_file = st.file_uploader(
                "上传图片",
                type=["png", "jpg", "jpeg"],
                key="attack_upload",
            )
            if uploaded_file is not None:
                image = Image.open(uploaded_file).convert("RGB")
                image_tensor = preprocess_custom_image(image)
                st.image(image, caption="上传图片", use_container_width=True)
                true_label = st.number_input("真实标签", min_value=0, max_value=999, value=0)
                if targeted:
                    target_label = st.number_input("目标标签", min_value=0, max_value=999, value=1)
        else:
            st.info("标准数据集将按批次生成对抗样本并自动保存实验记录。")
            num_samples = st.slider("样本数", 1, 100, 10)

    with col2:
        st.subheader("执行")
        if st.button("开始图像攻击", type="primary", use_container_width=True):
            if dataset_name == "自定义图片" and (uploaded_file is None or image_tensor is None or image is None):
                st.error("请先上传一张图片。")
                st.stop()

            with st.spinner("加载模型并执行攻击..."):
                model, checkpoint_path, checkpoint_status = resolve_classification_model(
                    model_name=model_name,
                    dataset_name=dataset_name,
                    device=device,
                )

                if dataset_name != "自定义图片" and not getattr(model, "fine_tuned_checkpoint_loaded", False):
                    st.error(
                        f"未找到与 {dataset_name} 匹配的微调权重：{checkpoint_path}。"
                        " 为保证实验有效性，标准数据集攻击已停止。"
                    )
                    st.code(
                        f"python scripts/train_classifier.py --dataset {dataset_to_key(dataset_name)} "
                        f"--model {model_name} --epochs 5",
                        language="bash",
                    )
                    st.stop()

                attacker = build_attacker(model, attack_method, attack_params, device)
                attack_name = attack_method.replace("&", "and").replace("/", "_")

                if dataset_name == "自定义图片":
                    labels = torch.tensor([int(true_label)])
                    target_tensor = (
                        torch.tensor([int(target_label)]) if targeted and target_label is not None else None
                    )
                    adv_images, info = attacker.generate(
                        image_tensor,
                        labels,
                        targeted=targeted,
                        target_labels=target_tensor,
                    )

                    with torch.no_grad():
                        orig_pred = model(image_tensor.to(device)).argmax(dim=1)
                        adv_pred = model(adv_images.to(device)).argmax(dim=1)

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
                        adv_pred.detach().cpu(),
                        attack_name,
                        metadata={
                            "model": model_name,
                            "dataset": dataset_name,
                            "targeted": targeted,
                            "target_label": int(target_label) if target_label is not None else None,
                            "metrics": sanitize_metrics(metrics),
                        },
                    )

                    store_last_experiment(
                        model_name=model_name,
                        dataset_name=dataset_name,
                        attack_method=attack_method,
                        metrics=metrics,
                        original=image_tensor.cpu(),
                        adversarial=adv_images.detach().cpu(),
                        labels=labels.cpu(),
                        orig_preds=orig_pred.cpu(),
                        adv_preds=adv_pred.cpu(),
                        save_dir=save_dir,
                        output_dim=int(model(image_tensor.to(device)).shape[1]),
                    )

                    st.success("对抗样本生成完成。")
                    p1, p2, p3 = st.columns(3)
                    with p1:
                        st.image(image, caption=f"原图（预测={orig_pred.item()}）", use_container_width=True)
                    with p2:
                        adv_np = adv_images[0].detach().cpu().permute(1, 2, 0).numpy()
                        st.image(np.clip(adv_np, 0, 1), caption=f"对抗图（预测={adv_pred.item()}）", use_container_width=True)
                    with p3:
                        pert = (adv_images[0].detach().cpu() - image_tensor[0].cpu()).abs().permute(1, 2, 0).numpy()
                        st.image(to_perturbation_vis(pert), caption="归一化扰动", use_container_width=True)

                    st.metric("攻击成功", "是" if orig_pred.item() != adv_pred.item() else "否")
                    st.write(f"L2 扰动：{info['perturbation_l2']:.4f}")
                    st.write(f"Linf 扰动：{info['perturbation_linf']:.4f}")
                    st.info(f"实验已保存到：{save_dir}")
                    show_metrics(metrics)
                else:
                    dataloader = data_manager.load_dataset(
                        dataset_to_key(dataset_name),
                        split="test",
                        batch_size=32,
                        shuffle=False,
                    )

                    all_original = []
                    all_adversarial = []
                    all_labels = []
                    all_orig_preds = []
                    all_adv_preds = []
                    sample_count = 0
                    output_dim = None

                    progress = st.progress(0.0)
                    status = st.empty()

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
                        progress.progress(min(sample_count / num_samples, 1.0))
                        status.text(f"已处理 {sample_count}/{num_samples} 个样本")

                    original = torch.cat(all_original)[:num_samples]
                    adversarial = torch.cat(all_adversarial)[:num_samples]
                    labels = torch.cat(all_labels)[:num_samples]
                    orig_preds = torch.cat(all_orig_preds)[:num_samples]
                    adv_preds = torch.cat(all_adv_preds)[:num_samples]

                    evaluator = AttackEvaluator(model, device=device)
                    metrics = evaluator.evaluate(original, adversarial, labels)

                    save_dir = data_manager.save_adversarial_samples(
                        original,
                        labels,
                        adversarial,
                        adv_preds,
                        attack_name,
                        metadata={
                            "model": model_name,
                            "dataset": dataset_name,
                            "targeted": targeted,
                            "num_samples": int(num_samples),
                            "metrics": sanitize_metrics(metrics),
                        },
                    )

                    store_last_experiment(
                        model_name=model_name,
                        dataset_name=dataset_name,
                        attack_method=attack_method,
                        metrics=metrics,
                        original=original,
                        adversarial=adversarial,
                        labels=labels,
                        orig_preds=orig_preds,
                        adv_preds=adv_preds,
                        save_dir=save_dir,
                        output_dim=output_dim,
                    )

                    st.success(f"已生成 {num_samples} 个对抗样本。")
                    st.info(f"实验已保存到：{save_dir}")
                    show_metrics(metrics)

                    d1, d2, d3 = st.columns(3)
                    with d1:
                        st.image(
                            np.clip(original[0].permute(1, 2, 0).numpy(), 0, 1),
                            caption=f"原图（标签={labels[0].item()}）",
                            use_container_width=True,
                        )
                    with d2:
                        st.image(
                            np.clip(adversarial[0].permute(1, 2, 0).numpy(), 0, 1),
                            caption=f"对抗图（预测={adv_preds[0].item()}）",
                            use_container_width=True,
                        )
                    with d3:
                        pert = (adversarial[0] - original[0]).abs().permute(1, 2, 0).numpy()
                        st.image(to_perturbation_vis(pert), caption="归一化扰动", use_container_width=True)


def render_results_tab(
    classification_models: list[str],
    device: str,
) -> None:
    st.header("结果分析")
    exp = st.session_state.get("last_experiment")
    if exp is None:
        st.info("请先在“攻击实验”页运行一次攻击。")
        return

    metrics = exp["metrics"]
    st.info(
        f"源模型：{exp['model']} | 数据集：{exp['dataset']} | 攻击：{exp['attack']} | 样本数：{exp['num_samples']}"
    )
    show_metrics(metrics)

    st.subheader("攻击前后对比")
    for idx in range(min(3, len(exp["original"]))):
        c1, c2, c3 = st.columns(3)
        with c1:
            st.image(
                np.clip(exp["original"][idx].permute(1, 2, 0).numpy(), 0, 1),
                caption=f"原图（标签={exp['labels'][idx].item()}）",
                use_container_width=True,
            )
        with c2:
            st.image(
                np.clip(exp["adversarial"][idx].permute(1, 2, 0).numpy(), 0, 1),
                caption=f"对抗图（预测={exp['adv_preds'][idx].item()}）",
                use_container_width=True,
            )
        with c3:
            pert = (exp["adversarial"][idx] - exp["original"][idx]).abs().permute(1, 2, 0).numpy()
            st.image(to_perturbation_vis(pert), caption="归一化扰动", use_container_width=True)

    st.subheader("黑盒迁移评测")
    target_options = [name for name in classification_models if name != exp["model"]]
    selected_targets = st.multiselect(
        "选择迁移评测目标模型",
        target_options,
        default=target_options[:1] if target_options else [],
    )

    if st.button("计算迁移率", use_container_width=True, key="transfer_eval_btn"):
        if not selected_targets:
            st.warning("请至少选择一个目标模型。")
        else:
            source_model, _, _ = resolve_classification_model(exp["model"], exp["dataset"], device)
            source_evaluator = AttackEvaluator(source_model, device=device)
            rows = []

            for target_model_name in selected_targets:
                target_model, checkpoint_path, _ = resolve_classification_model(
                    target_model_name,
                    exp["dataset"],
                    device,
                )
                if exp["dataset"] != "自定义图片" and not getattr(target_model, "fine_tuned_checkpoint_loaded", False):
                    st.warning(
                        f"跳过 {target_model_name}：未找到匹配 {exp['dataset']} 的微调权重 {checkpoint_path}。"
                    )
                    continue

                result = source_evaluator.evaluate_transferability(
                    exp["adversarial"].to(device),
                    exp["labels"].to(device),
                    [target_model],
                )
                rows.append(
                    {
                        "目标模型": target_model_name,
                        "迁移成功率(%)": round(float(result["average_transfer_rate"]), 2),
                    }
                )

            if rows:
                st.dataframe(pd.DataFrame(rows), use_container_width=True)
            else:
                st.info("没有可展示的迁移评测结果。")

    st.subheader("多模态迁移验证")
    clip_options = clip_model_choices()
    if not clip_options:
        st.info("当前环境未检测到可用的 CLIP 模型配置。")
    else:
        selected_clip_model = st.selectbox("选择 CLIP 模型", clip_options, key="clip_model_select")
        clip_prompt = st.text_input(
            "CLIP 文本提示",
            value="a photo of the main object",
            key="clip_prompt_input",
        )

        if st.button("计算 CLIP 相似度变化", use_container_width=True, key="clip_eval_btn"):
            with st.spinner("加载 CLIP 并计算相似度变化..."):
                clip_adapter = load_cached_evaluation_adapter("clip", selected_clip_model, device)
                clip_metrics = clip_adapter.evaluate(
                    exp["original"].to(device),
                    exp["adversarial"].to(device),
                    text_prompts=[clip_prompt] * int(exp["original"].shape[0]),
                )

            persist_multimodal_evaluation(exp, "clip", clip_metrics)

            c1, c2, c3 = st.columns(3)
            with c1:
                st.metric("特征余弦相似度", f"{clip_metrics['clip_cosine_similarity']:.4f}")
            with c2:
                st.metric("特征 L2 距离", f"{clip_metrics['clip_feature_l2']:.4f}")
            with c3:
                st.metric("图文相似度变化", f"{clip_metrics.get('text_similarity_change', 0.0):.4f}")

            if "text_image_similarity_orig" in clip_metrics and "text_image_similarity_adv" in clip_metrics:
                d1, d2 = st.columns(2)
                with d1:
                    st.metric("原图图文相似度", f"{clip_metrics['text_image_similarity_orig']:.4f}")
                with d2:
                    st.metric("对抗图图文相似度", f"{clip_metrics['text_image_similarity_adv']:.4f}")

            with st.expander("查看完整 CLIP 指标"):
                st.json(sanitize_metrics(clip_metrics))

    st.subheader("BLIP 图像描述对比")
    caption_options = caption_model_choices()
    if not caption_options:
        st.info("当前环境未检测到可用的 BLIP 图像描述模型。")
    else:
        selected_caption_model = st.selectbox("选择 BLIP 模型", caption_options, key="blip_model_select")
        caption_prompt = st.text_input(
            "描述提示词（可选）",
            value="",
            key="blip_prompt_input",
            help="留空时直接使用 BLIP 默认描述；填写后可引导描述风格。",
        )

        if st.button("计算原图描述 vs 对抗图描述", use_container_width=True, key="blip_eval_btn"):
            with st.spinner("加载 BLIP 并生成图像描述..."):
                blip_adapter = load_cached_evaluation_adapter("blip", selected_caption_model, device)
                blip_metrics = blip_adapter.evaluate(
                    exp["original"].to(device),
                    exp["adversarial"].to(device),
                    prompt=caption_prompt.strip() or None,
                )

            persist_multimodal_evaluation(exp, "blip", blip_metrics)

            c1, c2 = st.columns(2)
            with c1:
                st.metric("描述变化率", f"{blip_metrics['caption_change_rate']:.2f}%")
            with c2:
                st.metric("平均长度差", f"{blip_metrics['average_caption_length_delta']:.2f}")

            rows = []
            for idx, (orig_caption, adv_caption) in enumerate(
                zip(blip_metrics["original_captions"], blip_metrics["adversarial_captions"])
            ):
                rows.append(
                    {
                        "样本": idx,
                        "原图描述": orig_caption,
                        "对抗图描述": adv_caption,
                        "是否变化": "是" if orig_caption.strip() != adv_caption.strip() else "否",
                    }
                )

            st.dataframe(pd.DataFrame(rows), use_container_width=True)

            with st.expander("查看完整 BLIP 指标"):
                st.json(sanitize_metrics(blip_metrics))


def render_history_tab() -> None:
    st.header("历史记录")
    dm = DatasetManager()
    experiments = dm.list_experiments()
    if not experiments:
        st.info("暂无已保存的实验记录。")
        return

    records = []
    for exp_name in experiments:
        exp_path = os.path.join(dm.adversarial_dir, exp_name)
        try:
            metadata = dm.load_adversarial_samples(exp_path)
        except Exception:
            continue

        ts_raw = str(metadata.get("timestamp", ""))
        ts_dt = None
        try:
            ts_dt = datetime.strptime(ts_raw, "%Y%m%d_%H%M%S")
        except ValueError:
            pass

        records.append(
            {
                "name": exp_name,
                "path": exp_path,
                "attack": str(metadata.get("attack_name", "-")),
                "model": str(metadata.get("model", "-")),
                "dataset": str(metadata.get("dataset", "-")),
                "metadata": metadata,
                "timestamp": ts_dt,
                "timestamp_text": ts_dt.strftime("%Y-%m-%d %H:%M:%S") if ts_dt else ts_raw,
            }
        )

    if not records:
        st.info("历史目录存在，但没有可读取的实验记录。")
        return

    f1, f2, f3 = st.columns(3)
    with f1:
        attack_filter = st.multiselect(
            "攻击方法",
            sorted({record["attack"] for record in records}),
            default=sorted({record["attack"] for record in records}),
        )
    with f2:
        model_filter = st.multiselect(
            "模型",
            sorted({record["model"] for record in records}),
            default=sorted({record["model"] for record in records}),
        )
    with f3:
        dataset_filter = st.multiselect(
            "数据集",
            sorted({record["dataset"] for record in records}),
            default=sorted({record["dataset"] for record in records}),
        )

    filtered_records = [
        record
        for record in records
        if record["attack"] in attack_filter
        and record["model"] in model_filter
        and record["dataset"] in dataset_filter
    ]

    st.caption(f"筛选后实验数：{len(filtered_records)} / {len(records)}")
    if not filtered_records:
        st.warning("当前筛选条件下没有实验记录。")
        return

    experiment_rows = dm.build_experiment_index()
    if experiment_rows:
        exp_summary_df = pd.DataFrame(experiment_rows)
        if not exp_summary_df.empty:
            st.subheader("实验索引概览")
            if "target_family" in exp_summary_df.columns:
                chart_df = (
                    exp_summary_df.groupby(["task_type", "target_family"], dropna=False)
                    .size()
                    .reset_index(name="数量")
                )
            else:
                chart_df = exp_summary_df.groupby(["task_type"], dropna=False).size().reset_index(name="数量")
                chart_df["target_family"] = chart_df["task_type"]
            fig = px.bar(
                chart_df,
                x="target_family",
                y="数量",
                color="task_type",
                title="实验分布",
                template="plotly_dark",
            )
            fig.update_layout(margin=dict(l=20, r=20, t=50, b=20), legend_title_text="")
            st.plotly_chart(fig, use_container_width=True, key="history_index_chart")
            c1, c2 = st.columns(2)
            with c1:
                st.download_button(
                    "导出实验索引 CSV",
                    data=dm.export_experiment_index("csv"),
                    file_name="experiment_index.csv",
                    mime="text/csv",
                    use_container_width=True,
                    key="history_export_csv",
                )
            with c2:
                st.download_button(
                    "导出实验索引 JSON",
                    data=dm.export_experiment_index("json"),
                    file_name="experiment_index.json",
                    mime="application/json",
                    use_container_width=True,
                    key="history_export_json",
                )

    selected_name = st.selectbox(
        "选择实验",
        [record["name"] for record in filtered_records],
    )
    selected_record = next(record for record in filtered_records if record["name"] == selected_name)
    metadata = selected_record["metadata"]

    st.info(
        f"时间：{selected_record['timestamp_text']} | 攻击：{selected_record['attack']} | "
        f"模型：{selected_record['model']} | 数据集：{selected_record['dataset']}"
    )

    parsed_metrics = metadata.get("metrics", {})
    if parsed_metrics:
        if "original_accuracy" in parsed_metrics or "adversarial_accuracy" in parsed_metrics:
            c1, c2, c3 = st.columns(3)
            with c1:
                st.metric("原始准确率", f"{parsed_metrics.get('original_accuracy', 0.0):.2f}%")
            with c2:
                st.metric("攻击后准确率", f"{parsed_metrics.get('adversarial_accuracy', 0.0):.2f}%")
            with c3:
                st.metric("攻击成功率", f"{parsed_metrics.get('attack_success_rate', 0.0):.2f}%")
        else:
            c1, c2, c3 = st.columns(3)
            with c1:
                st.metric("扰动 L2", f"{parsed_metrics.get('perturbation_l2', 0.0):.4f}")
            with c2:
                st.metric("扰动 Linf", f"{parsed_metrics.get('perturbation_linf', 0.0):.4f}")
            with c3:
                st.metric("主任务指标", f"{parsed_metrics.get('similarity_drop', parsed_metrics.get('caption_change_rate', 0.0)):.4f}")

    decision = parsed_metrics.get("decision", {})
    if decision:
        if decision.get("attack_success"):
            st.success(f"统一判定：攻击成功。{decision.get('reason', '')}")
        else:
            st.warning(f"统一判定：攻击未达阈值。{decision.get('reason', '')}")

    multimodal_evaluations = metadata.get("multimodal_evaluations", {})
    if multimodal_evaluations:
        with st.expander("查看多模态迁移验证结果", expanded=False):
            clip_metrics = multimodal_evaluations.get("clip")
            if clip_metrics:
                st.markdown("**CLIP 图文对齐评测**")
                c1, c2, c3 = st.columns(3)
                with c1:
                    st.metric("余弦相似度", f"{clip_metrics.get('clip_cosine_similarity', 0.0):.4f}")
                with c2:
                    st.metric("特征 L2", f"{clip_metrics.get('clip_feature_l2', 0.0):.4f}")
                with c3:
                    st.metric("图文变化", f"{clip_metrics.get('text_similarity_change', 0.0):.4f}")

            blip_metrics = multimodal_evaluations.get("blip")
            if blip_metrics:
                st.markdown("**BLIP 图像描述评测**")
                d1, d2 = st.columns(2)
                with d1:
                    st.metric("描述变化率", f"{blip_metrics.get('caption_change_rate', 0.0):.2f}%")
                with d2:
                    st.metric("平均长度差", f"{blip_metrics.get('average_caption_length_delta', 0.0):.2f}")

                sample_rows = []
                orig_captions = blip_metrics.get("original_captions", [])
                adv_captions = blip_metrics.get("adversarial_captions", [])
                for idx, (orig_caption, adv_caption) in enumerate(zip(orig_captions, adv_captions)):
                    sample_rows.append(
                        {
                            "样本": idx,
                            "原图描述": orig_caption,
                            "对抗图描述": adv_caption,
                        }
                    )
                if sample_rows:
                    st.dataframe(pd.DataFrame(sample_rows), use_container_width=True)

    num_samples = int(metadata.get("num_samples", 0) or 0)
    if num_samples <= 0:
        num_samples = infer_saved_sample_count(selected_record["path"])

    if num_samples > 0:
        if num_samples == 1:
            sample_idx = 0
            st.caption("当前实验仅保存了 1 个样本。")
        else:
            sample_idx = st.slider("样本索引", 0, num_samples - 1, 0, 1)
        orig_path = os.path.join(selected_record["path"], f"original_{sample_idx}.png")
        adv_path = os.path.join(selected_record["path"], f"adversarial_{sample_idx}.png")
        pert_path = os.path.join(selected_record["path"], f"perturbation_{sample_idx}.png")

        c1, c2, c3 = st.columns(3)
        with c1:
            if os.path.exists(orig_path):
                st.image(orig_path, caption="原图", use_container_width=True)
            else:
                st.warning("未找到原图文件。")
        with c2:
            if os.path.exists(adv_path):
                st.image(adv_path, caption="对抗图", use_container_width=True)
            else:
                st.warning("未找到对抗图文件。")
        with c3:
            if os.path.exists(pert_path):
                st.image(pert_path, caption="扰动图", use_container_width=True)
            elif os.path.exists(orig_path) and os.path.exists(adv_path):
                orig_np = np.array(Image.open(orig_path).convert("RGB"), dtype=np.float32) / 255.0
                adv_np = np.array(Image.open(adv_path).convert("RGB"), dtype=np.float32) / 255.0
                st.image(to_perturbation_vis(np.abs(adv_np - orig_np)), caption="扰动图", use_container_width=True)
            else:
                st.warning("未找到扰动图文件。")
    else:
        st.info("当前实验未保存可预览的图像样本。")

    with st.expander("查看 metadata.json"):
        st.json(metadata)


def render_usage_tab() -> None:
    st.header("使用说明")
    reserved_adapters = "、".join(
        spec.display_name
        for spec in EvaluationAdapterRegistry.list_specs()
        if not spec.enabled
    )
    st.markdown(
        """
        - 平台主线已切换为“多模态目标优先”。
        - 默认攻击对象为 CLIP 图文对齐模型与 BLIP 图像描述模型。
        - 图像分类基线仍保留在代码中，但不再出现在默认主入口。
        - 攻击方法保留 FGSM / PGD，适合对多模态模型做图像扰动验证。
        - 结果分析页支持统一成功判定、CLIP 相似度、BLIP 描述变化和交叉评测。
        - 样本管理页支持上传、标注、版本快照与导出。
        - 历史记录页可回看实验配置、关键指标、成功判定与攻击前后样本对比。
        """
    )
    st.caption(f"预留适配器接口：{reserved_adapters}。当前版本仅保留接口，不启用云端依赖。")


st.set_page_config(
    page_title="多模态模型攻击与安全评估平台",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("多模态模型攻击与安全评估平台")
st.caption("主线版本：多模态目标优先（CLIP / BLIP）")
st.markdown("---")

clip_choices = clip_model_choices()
caption_choices = caption_model_choices()

st.sidebar.header("配置")
st.sidebar.caption("平台主线已切换为多模态目标优先，ResNet 不再出现在默认入口。")
attack_target_family = st.sidebar.selectbox("目标类型", ["CLIP 图文对齐", "BLIP 图像描述"], index=0)
available_models = clip_choices if attack_target_family == "CLIP 图文对齐" else caption_choices
if not available_models:
    st.sidebar.error("当前环境未检测到可用的目标模型配置。")
    st.stop()

model_name = st.sidebar.selectbox("目标模型", available_models, index=0)
attack_method = st.sidebar.selectbox("攻击方法", ["FGSM", "PGD"], index=1)
targeted = st.sidebar.checkbox("目标攻击", value=True)

st.sidebar.subheader("攻击参数")
if attack_method == "FGSM":
    attack_params = {"epsilon": st.sidebar.slider("epsilon", 0.0, 0.1, 8.0 / 255.0, 0.001)}
else:
    attack_params = {
        "epsilon": st.sidebar.slider("epsilon", 0.0, 0.1, 8.0 / 255.0, 0.001),
        "alpha": st.sidebar.slider("步长(alpha)", 0.0, 0.05, 2.0 / 255.0, 0.001),
        "num_steps": st.sidebar.slider("迭代次数", 1, 40, 10, 1),
    }

if attack_target_family == "BLIP 图像描述":
    st.sidebar.subheader("BLIP 强化参数")
    attack_params["num_restarts"] = st.sidebar.slider("随机重启次数", 1, 8, 3, 1)
    attack_params["source_weight"] = st.sidebar.slider("源描述抑制权重", 0.0, 2.0, 0.35, 0.05)

attack_mode = "image"

device = "cuda" if torch.cuda.is_available() else "cpu"
st.sidebar.info(f"设备：{device.upper()}")

tab1, tab2, tab3, tab4, tab5 = st.tabs(["多模态攻击", "结果分析", "样本管理", "历史记录", "使用说明"])

with tab1:
    render_multimodal_attack_tab(
        target_family=attack_target_family,
        model_name=model_name,
        attack_method=attack_method,
        attack_mode=attack_mode,
        attack_params=attack_params,
        targeted=targeted,
        device=device,
    )

with tab2:
    render_multimodal_results_tab(device=device)

with tab3:
    render_sample_management_tab()

with tab4:
    render_history_tab()

with tab5:
    render_usage_tab()

st.markdown("---")
st.caption("版本：Multimodal Target Priority")
