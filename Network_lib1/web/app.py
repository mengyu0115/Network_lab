# 1. 模型选择（移动到attack_method定义之后）
st.sidebar.subheader("1. 选择目标模型")

all_models = ModelLoader.get_supported_models()
vision_models = ['resnet18', 'resnet50', 'resnet101', 'vgg16', 'vgg19', 
                'densenet121', 'mobilenet_v2', 'efficientnet_b0']
clip_models = [m for m in all_models if 'clip' in m.lower()]
caption_models = [m for m in all_models if m not in vision_models and m not in clip_models]

# 根据攻击类型调整模型选项
if attack_method in ["Text-FGSM", "Text-PGD"]:
    # 文本攻击优先显示CLIP和caption模型
    model_options = []
    if clip_models:
        model_options.extend(["🔗 CLIP模型"] + clip_models)
    if caption_models:
        model_options.extend(["📝 图像描述模型"] + caption_models)
    model_options.extend(["📷 图像分类模型"] + vision_models)
else:
    # 图像攻击优先显示视觉模型
    model_options = ["📷 图像分类模型"] + vision_models
    if clip_models:
        model_options.extend(["🔗 CLIP模型"] + clip_models)
    if caption_models:
        model_options.extend(["📝 图像描述模型"] + caption_models)

model_name = st.sidebar.selectbox(
    "模型",
    model_options,
    index=0 if clip_models else 1
)

model_info = ModelLoader.get_model_info(model_name)
st.sidebar.caption(f"类型: {model_info.get('description', 'N/A')}")


# Tab 1: 攻击实验
with tab1:
    st.header("攻击实验")

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("输入配置")

        if dataset_name == "自定义图像":
            uploaded_file = st.file_uploader("上传图像", type=['png', 'jpg', 'jpeg'])

            if uploaded_file is not None:
                image = Image.open(uploaded_file).convert('RGB')
                st.image(image, caption="上传的图像", use_container_width=True)

                # 预处理
                from torchvision import transforms
                transform = transforms.Compose([
                    transforms.Resize(224),
                    transforms.ToTensor(),
                ])
                image_tensor = transform(image).unsqueeze(0)

                # 真实标签输入
                true_label = st.number_input("真实标签", min_value=0, max_value=999, value=0)

                if targeted:
                    target_label = st.number_input("目标标签", min_value=0, max_value=999, value=1)
                else:
                    target_label = None

        else:
            st.info("使用标准数据集进行批量测试")
            num_samples = st.slider("测试样本数量", 1, 100, 10)

    with col2:
        st.subheader("执行攻击")

        if st.button("🚀 开始攻击", type="primary", use_container_width=True):
            with st.spinner("正在加载模型..."):
                # 加载模型
                model = load_model(model_name, pretrained=True, device=device)
                st.success(f"✅ 模型 {model_name} 加载成功")

            with st.spinner("正在生成对抗样本..."):
                # 创建攻击器
                if attack_method == "FGSM":
                    attacker = FGSM(model, **attack_params, device=device)
                elif attack_method == "PGD":
                    attacker = PGD(model, **attack_params, device=device)
                elif attack_method == "C&W":
                    attacker = CarliniWagner(model, **attack_params, device=device)
                elif attack_method == "Text-FGSM":
                    attacker = TextFGSM(model, **attack_params, device=device)
                else:  # Text-PGD
                    attacker = TextPGD(model, **attack_params, device=device)

                # 检查是否是文本攻击
                is_text_attack = attack_method in ["Text-FGSM", "Text-PGD"]

                if is_text_attack:
                    # 文本攻击处理
                    st.info("请输入文本提示进行攻击（针对CLIP等多模态模型）")
                    
                    # 文本输入
                    text_prompt = st.text_input("文本提示", value="a photo of a cat")
                    true_label = st.number_input("真实标签", min_value=0, max_value=999, value=0)
                    
                    if targeted:
                        target_label = st.number_input("目标标签", min_value=0, max_value=999, value=1)
                    else:
                        target_label = None
                    
                    if st.button("生成文本嵌入并攻击", key="text_attack"):
                        
                        
                        
                        
                        
