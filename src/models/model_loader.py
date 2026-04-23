"""
Model Loader and Manager
模型加载与管理模块
"""
import os
import torch
import torch.nn as nn
import torchvision.models as models
from typing import Dict, Any, Optional, Tuple

try:
    from transformers import AutoModel, AutoTokenizer, CLIPProcessor, CLIPModel
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODEL_CACHE_ROOT = os.path.join(PROJECT_ROOT, 'data', 'model_cache')
TORCH_CACHE_DIR = os.path.join(MODEL_CACHE_ROOT, 'torch')
HF_CACHE_DIR = os.path.join(MODEL_CACHE_ROOT, 'huggingface')
CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, 'data', 'checkpoints')
LOCAL_CLIP_B32_DIR = os.path.join(HF_CACHE_DIR, 'clip-vit-base-patch32', '0_CLIPModel')

os.makedirs(TORCH_CACHE_DIR, exist_ok=True)
os.makedirs(HF_CACHE_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

os.environ.setdefault('TORCH_HOME', TORCH_CACHE_DIR)
os.environ.setdefault('HF_HOME', HF_CACHE_DIR)


class ImageCaptionModel(nn.Module):
    """图像描述模型包装器"""

    def __init__(self, base_model, processor=None, tokenizer=None):
        super().__init__()
        self.base_model = base_model
        self.processor = processor
        self.tokenizer = tokenizer
        self.model_type = 'caption'

    def forward(self, x):
        """返回分类 logits（用于攻击）"""
        return self.base_model(x)


class CLIPVisionModel(nn.Module):
    """CLIP视觉模型包装器"""

    def __init__(self, clip_model, processor=None, device='cpu'):
        super().__init__()
        self.clip_model = clip_model
        self.processor = processor
        self.device = device
        self.model_type = 'clip'

    def forward(self, x):
        """返回视觉特征 logits"""
        image_features = self.clip_model.get_image_features(x)
        return image_features


class ModelLoader:
    """模型加载器"""

    TORCHVISION_MODELS = {
        'resnet18': ('resnet18', 1000),
        'resnet50': ('resnet50', 1000),
    }

    CLIP_MODELS = {
        'clip ViT-B/32': 'openai/clip-vit-base-patch32',
        'clip ViT-L/14': 'openai/clip-vit-large-patch14',
    }

    CAPTION_MODELS = {
        'blip-base': 'Salesforce/blip-image-captioning-base',
        'git-base': 'microsoft/git-base',
    }
    DATASET_FINETUNE_CHECKPOINTS = {
        ('resnet18', 'cifar-10'): 'resnet18_cifar10.pth',
        ('resnet50', 'cifar-10'): 'resnet50_cifar10.pth',
        ('resnet18', 'mnist'): 'resnet18_mnist.pth',
        ('resnet50', 'mnist'): 'resnet50_mnist.pth',
    }

    @staticmethod
    def _normalize_dataset_name(dataset_name: Optional[str]) -> Optional[str]:
        if not dataset_name:
            return None
        ds = dataset_name.strip().lower()
        mapping = {
            'cifar10': 'cifar-10',
            'cifar-10': 'cifar-10',
            'mnist': 'mnist',
            'imagenet': 'imagenet',
            'image-net': 'imagenet',
        }
        return mapping.get(ds, ds)

    @staticmethod
    def get_finetune_checkpoint_path(model_name: str, dataset_name: Optional[str]) -> Optional[str]:
        ds = ModelLoader._normalize_dataset_name(dataset_name)
        if ds is None:
            return None
        filename = ModelLoader.DATASET_FINETUNE_CHECKPOINTS.get((model_name, ds))
        if not filename:
            return None
        return os.path.join(CHECKPOINT_DIR, filename)

    @staticmethod
    def _extract_state_dict(checkpoint_obj):
        if not isinstance(checkpoint_obj, dict):
            return None
        for key in ('state_dict', 'model_state_dict', 'net', 'model'):
            value = checkpoint_obj.get(key)
            if isinstance(value, dict):
                return value
        if all(isinstance(v, torch.Tensor) for v in checkpoint_obj.values()):
            return checkpoint_obj
        return None

    @staticmethod
    def maybe_load_finetune_checkpoint(model: nn.Module,
                                       model_name: str,
                                       dataset_name: Optional[str],
                                       device: str) -> Tuple[bool, Optional[str], str]:
        checkpoint_path = ModelLoader.get_finetune_checkpoint_path(model_name, dataset_name)
        if not checkpoint_path:
            return False, None, "no_checkpoint_mapping"
        if not os.path.exists(checkpoint_path):
            return False, checkpoint_path, "checkpoint_not_found"

        checkpoint_obj = torch.load(checkpoint_path, map_location=device)
        state_dict = ModelLoader._extract_state_dict(checkpoint_obj)
        if state_dict is None:
            return False, checkpoint_path, "invalid_checkpoint_format"

        cleaned = {}
        for key, value in state_dict.items():
            new_key = key[7:] if key.startswith('module.') else key
            cleaned[new_key] = value

        model.load_state_dict(cleaned, strict=False)
        return True, checkpoint_path, "loaded"

    @staticmethod
    def get_clip_source(model_name: str) -> str:
        """Resolve CLIP model source: prefer local directory when available."""
        if model_name == 'clip ViT-B/32' and os.path.isdir(LOCAL_CLIP_B32_DIR):
            return LOCAL_CLIP_B32_DIR
        return ModelLoader.CLIP_MODELS[model_name]

    @staticmethod
    def load_model(model_name: str, pretrained: bool = True,
                   num_classes: int = 1000, device: str = 'cuda',
                   dataset_name: Optional[str] = None) -> nn.Module:
        """
        加载预训练模型

        Args:
            model_name: 模型名称
            pretrained: 是否加载预训练权重
            num_classes: 分类数量
            device: 计算设备

        Returns:
            model: 加载的模型
        """
        device = device if torch.cuda.is_available() else 'cpu'

        if model_name in ModelLoader.TORCHVISION_MODELS:
            model = ModelLoader._load_torchvision_model(model_name, pretrained, num_classes, device)
            loaded, checkpoint_path, load_status = ModelLoader.maybe_load_finetune_checkpoint(
                model=model,
                model_name=model_name,
                dataset_name=dataset_name,
                device=device,
            )
            model.fine_tuned_checkpoint_loaded = loaded
            model.fine_tuned_checkpoint_path = checkpoint_path
            model.fine_tuned_checkpoint_status = load_status
            model.selected_dataset_name = dataset_name
            return model
        elif model_name in ModelLoader.CLIP_MODELS:
            return ModelLoader._load_clip_model(model_name, pretrained, device)
        elif model_name in ModelLoader.CAPTION_MODELS:
            raise NotImplementedError(
                "Caption generation models are not supported in the classification attack pipeline yet. "
                "Use a torchvision classification model or a CLIP vision model."
            )
        else:
            raise ValueError(f"不支持的模型: {model_name}. 支持的模型: {ModelLoader.get_supported_models()}")

    @staticmethod
    def _load_torchvision_model(model_name: str, pretrained: bool,
                               num_classes: int, device: str) -> nn.Module:
        """加载 torchvision 模型"""
        model_fn_name = ModelLoader.TORCHVISION_MODELS[model_name][0]
        model_fn = getattr(models, model_fn_name)

        weights = 'DEFAULT' if pretrained else None
        model = model_fn(weights=weights)

        if num_classes != 1000:
            if 'resnet' in model_name:
                in_features = model.fc.in_features
                model.fc = nn.Linear(in_features, num_classes)
            elif 'densenet' in model_name:
                in_features = model.classifier.in_features
                model.classifier = nn.Linear(in_features, num_classes)
            elif 'vgg' in model_name:
                in_features = model.classifier[6].in_features
                model.classifier[6] = nn.Linear(in_features, num_classes)
            elif 'mobilenet' in model_name or 'efficientnet' in model_name:
                in_features = model.classifier[1].in_features
                model.classifier[1] = nn.Linear(in_features, num_classes)

        model = model.to(device)
        model.eval()
        return model

    @staticmethod
    def _load_clip_model(model_name: str, pretrained: bool, device: str) -> nn.Module:
        """加载 CLIP 模型"""
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("需要安装 transformers 库: pip install transformers")

        model_path = ModelLoader.get_clip_source(model_name)
        if os.path.isdir(model_path):
            clip_model = CLIPModel.from_pretrained(model_path, local_files_only=True).to(device)
            processor = CLIPProcessor.from_pretrained(model_path, local_files_only=True)
        else:
            clip_model = CLIPModel.from_pretrained(model_path, cache_dir=HF_CACHE_DIR).to(device)
            processor = CLIPProcessor.from_pretrained(model_path, cache_dir=HF_CACHE_DIR)
        clip_model.eval()

        return CLIPVisionModel(clip_model, processor=processor, device=device)

    @staticmethod
    def _load_caption_model(model_name: str, pretrained: bool, device: str) -> nn.Module:
        """加载图像描述模型"""
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("需要安装 transformers 库: pip install transformers")

        from transformers import BlipForConditionalGeneration, GITForCausalLM

        model_path = ModelLoader.CAPTION_MODELS[model_name]

        if 'blip' in model_name:
            base_model = BlipForConditionalGeneration.from_pretrained(model_path, cache_dir=HF_CACHE_DIR)
        elif 'git' in model_name:
            base_model = GITForCausalLM.from_pretrained(model_path, cache_dir=HF_CACHE_DIR)
        else:
            base_model = AutoModel.from_pretrained(model_path, cache_dir=HF_CACHE_DIR)

        base_model = base_model.to(device)
        base_model.eval()

        return ImageCaptionModel(base_model)

    @staticmethod
    def get_supported_models(include_clip: bool = True, include_caption: bool = False) -> list:
        """获取所有支持的模型列表"""
        models = list(ModelLoader.TORCHVISION_MODELS.keys())
        if TRANSFORMERS_AVAILABLE and include_clip:
            models.extend(ModelLoader.CLIP_MODELS.keys())
        if TRANSFORMERS_AVAILABLE and include_caption:
            models.extend(ModelLoader.CAPTION_MODELS.keys())
        return sorted(models)

    @staticmethod
    def get_model_info(model_name: str) -> Dict[str, Any]:
        """获取模型信息"""
        info = {'type': 'torchvision', 'num_classes': 1000}

        if model_name in ModelLoader.TORCHVISION_MODELS:
            _, num_classes = ModelLoader.TORCHVISION_MODELS[model_name]
            info = {'type': 'vision', 'num_classes': num_classes, 'description': '图像分类模型'}
        elif model_name in ModelLoader.CLIP_MODELS:
            info = {'type': 'clip', 'num_classes': 512, 'description': 'CLIP视觉模型'}
        elif model_name in ModelLoader.CAPTION_MODELS:
            info = {'type': 'caption', 'num_classes': 'text', 'description': '图像描述模型'}

        return info


def load_model(model_name: str, **kwargs) -> nn.Module:
    """便捷函数：加载模型"""
    return ModelLoader.load_model(model_name, **kwargs)
