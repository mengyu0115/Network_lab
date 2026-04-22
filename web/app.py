"""
多模态对抗攻击平台 - Streamlit 页面
"""
import os
import sys
from typing import List

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.attacks import (
    FGSM,
    PGD,
    CarliniWagner,
    TextFGSM,
    TextPGD,
    CLIPMultimodalAttack,
)
from src.data_manager import DatasetManager
from src.evaluation import AttackEvaluator, CLIPMultimodalEvaluator
from src.models import ModelLoader, load_model


class PrototypeTextClassifier(nn.Module):
    """用原型文本做余弦相似度分类。"""

    def __init__(self, prototypes: torch.Tensor):
        super().__init__()
        self.register_buffer("prototypes", F.normalize(prototypes, p=2, dim=-1))

    def forward(self, text_embeddings: torch.Tensor) -> torch.Tensor:
        emb = F.normalize(text_embeddings, p=2, dim=-1)
        return emb @ self.prototypes.t()


def build_text_embeddings(text_model, tokenizer, texts: List[str], device: str) -> torch.Tensor:
    inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = text_model(**inputs)
        if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
            return outputs.pooler_output
        return outputs.last_hidden_state.mean(dim=1)


def to_perturbation_vis(perturbation: np.ndarray) -> np.ndarray:
    max_val = float(perturbation.max())
    if max_val <= 1e-12:
        return perturbation
    return np.clip(perturbation / max_val, 0.0, 1.0)


@st.cache_resource(show_spinner=False)
def get_cached_vision_model(model_name: str, device: str, num_classes: int = 1000):
    return load_model(model_name, pretrained=True, device=device, num_classes=num_classes)


@st.cache_resource(show_spinner=False)
def get_cached_clip_text_stack(model_name: str, device: str):
    from transformers import CLIPTextModel, CLIPTokenizer

    clip_source = ModelLoader.get_clip_source(model_name)
    if os.path.isdir(clip_source):
        tokenizer = CLIPTokenizer.from_pretrained(clip_source, local_files_only=True)
        text_model = CLIPTextModel.from_pretrained(clip_source, local_files_only=True).to(device)
    else:
        tokenizer = CLIPTokenizer.from_pretrained(clip_source)
        text_model = CLIPTextModel.from_pretrained(clip_source).to(device)
    text_model.eval()
    return tokenizer, text_model


@st.cache_resource(show_spinner=False)
def get_cached_clip_model_stack(model_name: str, device: str):
    from transformers import CLIPModel, CLIPProcessor

    clip_source = ModelLoader.get_clip_source(model_name)
    if os.path.isdir(clip_source):
        clip_model = CLIPModel.from_pretrained(clip_source, local_files_only=True).to(device)
        processor = CLIPProcessor.from_pretrained(clip_source, local_files_only=True)
    else:
        clip_model = CLIPModel.from_pretrained(clip_source).to(device)
        processor = CLIPProcessor.from_pretrained(clip_source)
    clip_model.eval()
    return clip_model, processor


st.set_page_config(
    page_title="多模态对抗攻击平台",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("🛡️ 多模态对抗攻击平台")
st.markdown("---")

st.sidebar.header("配置")
attack_method = st.sidebar.selectbox(
    "攻击方法",
    ["FGSM", "PGD", "C&W", "Text-FGSM", "Text-PGD", "MM-CLIP"],
    index=0,
)
is_text_attack = attack_method in ["Text-FGSM", "Text-PGD"]
is_multimodal_attack = attack_method == "MM-CLIP"

if is_text_attack or is_multimodal_attack:
    clip_model_names = list(ModelLoader.CLIP_MODELS.keys())
    model_name = st.sidebar.selectbox("CLIP模型", clip_model_names, index=0)
    dataset_name = "图文输入"
else:
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
elif attack_method == "Text-FGSM":
    attack_params = {"epsilon": st.sidebar.slider("文本epsilon", 0.0, 1.0, 0.1, 0.01)}
elif attack_method == "Text-PGD":
    attack_params = {
        "epsilon": st.sidebar.slider("文本epsilon", 0.0, 1.0, 0.1, 0.01),
        "alpha": st.sidebar.slider("文本步长", 0.001, 0.1, 0.01, 0.001),
        "num_iter": st.sidebar.slider("迭代次数", 5, 50, 10, 5),
    }
else:
    mm_mode = st.sidebar.selectbox("多模态攻击模式", ["image", "text", "joint"], index=2)
    attack_params = {
        "image_epsilon": st.sidebar.slider("图像epsilon", 0.0, 0.1, 8.0 / 255.0, 0.001),
        "image_alpha": st.sidebar.slider("图像步长", 0.0005, 0.02, 1.0 / 255.0, 0.0005),
        "text_epsilon": st.sidebar.slider("文本特征epsilon", 0.0, 1.0, 0.20, 0.01),
        "text_alpha": st.sidebar.slider("文本特征步长", 0.001, 0.1, 0.02, 0.001),
        "num_steps": st.sidebar.slider("迭代次数", 5, 80, 20, 5),
        "mode": mm_mode,
    }

targeted = st.sidebar.checkbox("目标攻击", value=False)
device = "cuda" if torch.cuda.is_available() else "cpu"
st.sidebar.info(f"设备：{device.upper()}")

tab1, tab2, tab3, tab4 = st.tabs(["攻击实验", "结果分析", "历史记录", "使用说明"])

with tab1:
    st.header("运行攻击")
    col1, col2 = st.columns([1, 1])

    if is_multimodal_attack:
        with col1:
            st.subheader("图文输入")
            uploaded_file = st.file_uploader("上传图片", type=["png", "jpg", "jpeg"], key="mm_img")
            source_text = st.text_input("源文本", value="一只猫在蓝色沙发上", key="mm_src")
            target_text = st.text_input("目标文本", value="一辆汽车停在路边", key="mm_tgt")

            image_tensor = None
            if uploaded_file is not None:
                image = Image.open(uploaded_file).convert("RGB")
                st.image(image, caption="输入图片", use_container_width=True)
                from torchvision import transforms

                tfm = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
                image_tensor = tfm(image).unsqueeze(0)

        with col2:
            st.subheader("执行")
            if st.button("开始多模态攻击", type="primary", use_container_width=True):
                if image_tensor is None:
                    st.error("请先上传图片。")
                else:
                    with st.spinner("加载CLIP模型..."):
                        clip_model, clip_processor = get_cached_clip_model_stack(model_name, device)

                    attacker = CLIPMultimodalAttack(
                        clip_model=clip_model,
                        processor=clip_processor,
                        device=device,
                        image_epsilon=attack_params["image_epsilon"],
                        image_alpha=attack_params["image_alpha"],
                        text_epsilon=attack_params["text_epsilon"],
                        text_alpha=attack_params["text_alpha"],
                        num_steps=attack_params["num_steps"],
                    )
                    evaluator = CLIPMultimodalEvaluator(clip_model, clip_processor, device=device)

                    with st.spinner("执行多模态攻击优化..."):
                        adv_image, adv_text_feat, info = attacker.generate(
                            image=image_tensor,
                            source_text=source_text,
                            target_text=target_text if targeted else None,
                            mode=attack_params["mode"],
                            targeted=targeted,
                        )
                        mm_metrics = evaluator.evaluate(
                            original_image=image_tensor.to(device),
                            adversarial_image=adv_image,
                            source_text=source_text,
                            target_text=target_text if targeted else None,
                            adversarial_text_feature=adv_text_feat,
                            targeted=targeted,
                        )

                    st.success("多模态攻击完成。")
                    c1, c2, c3 = st.columns(3)
                    with c1:
                        st.metric("源相似度(原图)", f"{mm_metrics['orig_source_similarity']:.4f}")
                    with c2:
                        st.metric("源相似度(对抗)", f"{mm_metrics['adv_source_similarity']:.4f}")
                    with c3:
                        st.metric("相似度下降", f"{mm_metrics['similarity_drop']:.4f}")

                    if targeted:
                        t1, t2 = st.columns(2)
                        with t1:
                            st.metric("目标相似度增益", f"{mm_metrics['target_similarity_gain']:.4f}")
                        with t2:
                            st.metric("目标攻击成功", "是" if mm_metrics["targeted_attack_success"] else "否")

                    i1, i2 = st.columns(2)
                    with i1:
                        orig_img = image_tensor[0].detach().cpu().permute(1, 2, 0).numpy()
                        st.image(np.clip(orig_img, 0, 1), caption="原图", use_container_width=True)
                    with i2:
                        adv_img = adv_image[0].detach().cpu().permute(1, 2, 0).numpy()
                        st.image(np.clip(adv_img, 0, 1), caption="对抗图", use_container_width=True)

                    pert = (adv_image[0] - image_tensor[0].to(device)).abs().detach().cpu().permute(1, 2, 0).numpy()
                    st.image(to_perturbation_vis(pert), caption="扰动(归一化)", use_container_width=True)

                    st.session_state["last_mm_experiment"] = {
                        "model": model_name,
                        "source_text": source_text,
                        "target_text": target_text if targeted else "",
                        "mode": attack_params["mode"],
                        "targeted": targeted,
                        "metrics": mm_metrics,
                        "info": info,
                        "orig_img": image_tensor[0].detach().cpu(),
                        "adv_img": adv_image[0].detach().cpu(),
                    }

    elif is_text_attack:
        with col1:
            st.subheader("文本输入")
            text_prompt = st.text_input("待攻击提示词", value="一只猫在沙发上")
            prototype_text = st.text_area(
                "类别原型（每行一个）",
                value="一只猫在沙发上\n一只狗在草地上\n一辆汽车在路上",
                height=120,
            )
            class_prompts = [p.strip() for p in prototype_text.splitlines() if p.strip()]
            if len(class_prompts) < 2:
                st.warning("请至少提供两个类别原型。")

            true_label = st.number_input("真实类别索引", min_value=0, max_value=max(len(class_prompts) - 1, 0), value=0, step=1)
            target_label = None
            if targeted:
                target_label = st.number_input("目标类别索引", min_value=0, max_value=max(len(class_prompts) - 1, 0), value=min(1, max(len(class_prompts) - 1, 0)), step=1)

        with col2:
            st.subheader("执行")
            if st.button("开始文本攻击", type="primary", use_container_width=True):
                if len(class_prompts) < 2:
                    st.error("至少需要两个类别原型。")
                else:
                    with st.spinner("加载CLIP文本编码器..."):
                        tokenizer, text_model = get_cached_clip_text_stack(model_name, device)

                    with st.spinner("构建文本嵌入..."):
                        proto_embeddings = build_text_embeddings(text_model, tokenizer, class_prompts, device)
                        sample_embedding = build_text_embeddings(text_model, tokenizer, [text_prompt], device)

                    classifier = PrototypeTextClassifier(proto_embeddings).to(device).eval()
                    attacker = TextFGSM(classifier, **attack_params, device=device) if attack_method == "Text-FGSM" else TextPGD(classifier, **attack_params, device=device)
                    target_tensor = torch.tensor([int(target_label)], device=device) if targeted and target_label is not None else None
                    adv_embedding, info = attacker.generate(
                        sample_embedding,
                        torch.tensor([int(true_label)], device=device),
                        targeted=targeted,
                        target_labels=target_tensor,
                    )

                    with torch.no_grad():
                        orig_pred = classifier(sample_embedding).argmax(dim=1).item()
                        adv_pred = classifier(adv_embedding).argmax(dim=1).item()

                    st.success("文本攻击完成。")
                    st.metric("原始预测", class_prompts[orig_pred])
                    st.metric("对抗预测", class_prompts[adv_pred])
                    st.metric("攻击成功", "是" if orig_pred != adv_pred else "否")
                    st.write(f"L2扰动：{info['perturbation_l2']:.4f}")
                    st.write(f"Linf扰动：{info['perturbation_linf']:.4f}")

                    fig, ax = plt.subplots(figsize=(8, 4))
                    ax.plot(sample_embedding[0].detach().cpu().numpy(), label="原始嵌入")
                    ax.plot(adv_embedding[0].detach().cpu().numpy(), label="对抗嵌入")
                    ax.set_title("嵌入对比")
                    ax.set_xlabel("维度")
                    ax.set_ylabel("数值")
                    ax.legend()
                    st.pyplot(fig)
    else:
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
                with st.spinner("加载模型（首次会下载，后续复用缓存）..."):
                    dataset_num_classes = {"CIFAR-10": 10, "MNIST": 10}.get(dataset_name, 1000)
                    model = get_cached_vision_model(model_name, device, dataset_num_classes)
                    if dataset_num_classes != 1000:
                        st.info(
                            f"已将分类头改为 {dataset_num_classes} 类以匹配 {dataset_name}。"
                            " 该分类头未在该数据集微调，建议后续接入微调权重。"
                        )

                if attack_method == "FGSM":
                    attacker = FGSM(model, **attack_params, device=device)
                elif attack_method == "PGD":
                    attacker = PGD(model, **attack_params, device=device)
                else:
                    attacker = CarliniWagner(model, **attack_params, device=device)

                if dataset_name == "自定义图片" and uploaded_file is not None and image_tensor is not None:
                    labels = torch.tensor([int(true_label)])
                    target_tensor = torch.tensor([int(target_label)]) if targeted and target_label is not None else None
                    adv_images, info = attacker.generate(image_tensor, labels, targeted=targeted, target_labels=target_tensor)

                    with torch.no_grad():
                        orig_pred = model(image_tensor.to(device)).argmax(dim=1).item()
                        adv_pred = model(adv_images.to(device)).argmax(dim=1).item()

                    st.success("对抗样本生成完成。")
                    c1, c2, c3 = st.columns(3)
                    with c1:
                        st.image(image, caption=f"原图(预测={orig_pred})", use_container_width=True)
                    with c2:
                        adv_img = adv_images[0].detach().cpu().permute(1, 2, 0).numpy()
                        st.image(np.clip(adv_img, 0, 1), caption=f"对抗图(预测={adv_pred})", use_container_width=True)
                    with c3:
                        pert = (adv_images[0] - image_tensor[0]).abs().detach().cpu().permute(1, 2, 0).numpy()
                        st.image(to_perturbation_vis(pert), caption="扰动(归一化)", use_container_width=True)

                    st.metric("攻击成功", "是" if orig_pred != adv_pred else "否")
                    st.write(f"L2扰动：{info['perturbation_l2']:.4f}")
                    st.write(f"Linf扰动：{info['perturbation_linf']:.4f}")
                else:
                    data_manager = DatasetManager()
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

                    c1, c2, c3, c4 = st.columns(4)
                    with c1:
                        st.metric("预测翻转率", f"{metrics['attack_success_rate']:.2f}%")
                    with c2:
                        st.metric("L2扰动", f"{metrics['perturbation_l2']:.4f}")
                    with c3:
                        st.metric("Linf扰动", f"{metrics['perturbation_linf']:.4f}")
                    with c4:
                        st.metric("SSIM", f"{metrics['ssim']:.4f}")

with tab2:
    st.header("结果分析")
    if "last_mm_experiment" in st.session_state:
        mm = st.session_state["last_mm_experiment"]
        st.subheader("多模态(CLIP)攻击结果")
        st.info(f"模型：{mm['model']} | 模式：{mm['mode']} | 目标攻击：{'是' if mm['targeted'] else '否'}")
        m = mm["metrics"]
        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("源相似度(原图)", f"{m['orig_source_similarity']:.4f}")
        with c2:
            st.metric("源相似度(对抗)", f"{m['adv_source_similarity']:.4f}")
        with c3:
            st.metric("相似度下降", f"{m['similarity_drop']:.4f}")
        if mm["targeted"]:
            t1, t2 = st.columns(2)
            with t1:
                st.metric("目标相似度增益", f"{m['target_similarity_gain']:.4f}")
            with t2:
                st.metric("目标攻击成功", "是" if m["targeted_attack_success"] else "否")

        i1, i2, i3 = st.columns(3)
        with i1:
            st.image(np.clip(mm["orig_img"].permute(1, 2, 0).numpy(), 0, 1), caption="原图", use_container_width=True)
        with i2:
            st.image(np.clip(mm["adv_img"].permute(1, 2, 0).numpy(), 0, 1), caption="对抗图", use_container_width=True)
        with i3:
            pert = (mm["adv_img"] - mm["orig_img"]).abs().permute(1, 2, 0).numpy()
            st.image(to_perturbation_vis(pert), caption="扰动(归一化)", use_container_width=True)
        st.markdown("---")

    if "last_experiment" not in st.session_state:
        st.info("请先运行一次图像攻击实验，或运行一次多模态攻击实验。")
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

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.metric("预测翻转率", f"{metrics['attack_success_rate']:.2f}%")
        with c2:
            st.metric("L2扰动", f"{metrics['perturbation_l2']:.4f}")
        with c3:
            st.metric("Linf扰动", f"{metrics['perturbation_linf']:.4f}")
        with c4:
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
        selected = st.selectbox("选择实验", experiments)
        if selected:
            exp_path = os.path.join(dm.adversarial_dir, selected)
            metadata = dm.load_adversarial_samples(exp_path)
            st.json(metadata)

with tab4:
    st.header("使用说明")
    st.markdown(
        """
        - 图像攻击：FGSM / PGD / C&W，用于分类模型鲁棒性测试。
        - 文本攻击：在CLIP文本嵌入空间上进行FGSM/PGD扰动。
        - 多模态攻击(MM-CLIP)：对图像-文本对齐做 image/text/joint 三种攻击。
        - 多模态评估指标包括源文本相似度下降、目标文本相似度增益（目标攻击时）。
        """
    )

st.markdown("---")
st.caption("多模态对抗攻击平台 v1.2")
