"""
Data Manager Module
数据集管理模块
"""
import os
import json
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets
from typing import List, Tuple, Optional, Dict
import shutil
from datetime import datetime


class DatasetManager:
    """数据集管理器"""

    def __init__(self, data_root: str = './data'):
        """
        Args:
            data_root: 数据根目录
        """
        self.data_root = data_root
        self.raw_dir = os.path.join(data_root, 'raw')
        self.processed_dir = os.path.join(data_root, 'processed')
        self.adversarial_dir = os.path.join(data_root, 'adversarial')

        os.makedirs(self.raw_dir, exist_ok=True)
        os.makedirs(self.processed_dir, exist_ok=True)
        os.makedirs(self.adversarial_dir, exist_ok=True)

    def load_dataset(self, dataset_name: str, split: str = 'test',
                     batch_size: int = 32, shuffle: bool = False) -> DataLoader:
        """
        加载标准数据集

        Args:
            dataset_name: 数据集名称 (cifar10, imagenet, mnist等)
            split: 数据集划分
            batch_size: 批次大小
            shuffle: 是否打乱

        Returns:
            dataloader: 数据加载器
        """
        # 根据数据集选择不同的 transform
        if dataset_name.lower() == 'mnist':
            # MNIST 是灰度图，需要转为 3 通道
            transform = transforms.Compose([
                transforms.Resize(224),
                transforms.Grayscale(num_output_channels=3),
                transforms.ToTensor(),
            ])
        else:
            transform = transforms.Compose([
                transforms.Resize(224),
                transforms.ToTensor(),
            ])

        if dataset_name.lower() == 'cifar10':
            dataset = datasets.CIFAR10(
                root=self.raw_dir,
                train=(split == 'train'),
                download=True,
                transform=transform
            )
        elif dataset_name.lower() == 'mnist':
            dataset = datasets.MNIST(
                root=self.raw_dir,
                train=(split == 'train'),
                download=True,
                transform=transform
            )
        elif dataset_name.lower() == 'imagenet':
            dataset = datasets.ImageFolder(
                root=os.path.join(self.raw_dir, 'imagenet', split),
                transform=transform
            )
        else:
            raise ValueError(f"不支持的数据集: {dataset_name}")

        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=2
        )

        return dataloader

    def save_adversarial_samples(self, images: torch.Tensor, labels: torch.Tensor,
                                 adv_images: torch.Tensor, predictions: torch.Tensor,
                                 attack_name: str, metadata: Dict) -> str:
        """
        保存对抗样本

        Args:
            images: 原始图像
            labels: 真实标签
            adv_images: 对抗样本
            predictions: 模型预测
            attack_name: 攻击名称
            metadata: 元数据

        Returns:
            save_path: 保存路径
        """
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        save_dir = os.path.join(self.adversarial_dir, f"{attack_name}_{timestamp}")
        os.makedirs(save_dir, exist_ok=True)

        # 保存图像
        for i in range(len(images)):
            # 原始图像
            orig_path = os.path.join(save_dir, f"original_{i}.png")
            self._save_image(images[i], orig_path)

            # 对抗样本
            adv_path = os.path.join(save_dir, f"adversarial_{i}.png")
            self._save_image(adv_images[i], adv_path)

            # 扰动可视化
            perturbation = (adv_images[i] - images[i]).abs()
            pert_path = os.path.join(save_dir, f"perturbation_{i}.png")
            self._save_image(perturbation, pert_path)

        # 保存元数据
        metadata_full = {
            'attack_name': attack_name,
            'timestamp': timestamp,
            'num_samples': len(images),
            'labels': labels.cpu().tolist(),
            'predictions': predictions.cpu().tolist(),
            **metadata
        }

        metadata_path = os.path.join(save_dir, 'metadata.json')
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata_full, f, indent=2, ensure_ascii=False)

        return save_dir

    def _save_image(self, tensor: torch.Tensor, path: str):
        """保存张量为图像"""
        if tensor.dim() == 3:
            tensor = tensor.cpu().detach()
            if tensor.shape[0] == 1:
                tensor = tensor.repeat(3, 1, 1)
            img = transforms.ToPILImage()(tensor)
            img.save(path)

    def load_adversarial_samples(self, experiment_dir: str) -> Dict:
        """
        加载对抗样本实验结果

        Args:
            experiment_dir: 实验目录

        Returns:
            data: 包含图像和元数据的字典
        """
        metadata_path = os.path.join(experiment_dir, 'metadata.json')
        with open(metadata_path, 'r', encoding='utf-8') as f:
            metadata = json.load(f)

        return metadata

    def list_experiments(self) -> List[str]:
        """列出所有实验"""
        experiments = []
        if os.path.exists(self.adversarial_dir):
            experiments = [d for d in os.listdir(self.adversarial_dir)
                          if os.path.isdir(os.path.join(self.adversarial_dir, d))]
        return sorted(experiments, reverse=True)

    def filter_samples(self, image_paths: List[str], labels: Optional[List[int]] = None,
                      include_labels: Optional[List[int]] = None,
                      exclude_labels: Optional[List[int]] = None,
                      max_samples: Optional[int] = None) -> Tuple[List[str], Optional[List[int]]]:
        """Filter samples by label constraints."""
        if labels is not None and len(image_paths) != len(labels):
            raise ValueError("image_paths and labels must have the same length")

        include_set = set(include_labels) if include_labels else None
        exclude_set = set(exclude_labels) if exclude_labels else set()

        kept_indices = []
        for idx in range(len(image_paths)):
            if labels is None:
                kept_indices.append(idx)
                continue

            label = labels[idx]
            if include_set is not None and label not in include_set:
                continue
            if label in exclude_set:
                continue
            kept_indices.append(idx)

        if max_samples is not None:
            kept_indices = kept_indices[:max_samples]

        filtered_paths = [image_paths[i] for i in kept_indices]
        filtered_labels = [labels[i] for i in kept_indices] if labels is not None else None
        return filtered_paths, filtered_labels

    def update_annotations(self, annotations: Dict[str, Dict],
                          annotation_file: str = 'annotations.json',
                          merge: bool = True) -> str:
        """Create or update annotation records in processed directory."""
        annotation_path = os.path.join(self.processed_dir, annotation_file)
        os.makedirs(os.path.dirname(annotation_path), exist_ok=True)

        current = {}
        if merge and os.path.exists(annotation_path):
            with open(annotation_path, 'r', encoding='utf-8') as f:
                current = json.load(f)

        if merge:
            for key, value in annotations.items():
                if key not in current or not isinstance(current.get(key), dict):
                    current[key] = {}
                if isinstance(value, dict):
                    current[key].update(value)
                else:
                    current[key]['value'] = value
            payload = current
        else:
            payload = annotations

        with open(annotation_path, 'w', encoding='utf-8') as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)

        return annotation_path

    def create_dataset_version(self, source_dir: str,
                              version_name: Optional[str] = None,
                              metadata: Optional[Dict] = None) -> str:
        """Create an immutable dataset snapshot under processed/versions."""
        if not os.path.exists(source_dir):
            raise FileNotFoundError(f"source_dir not found: {source_dir}")

        versions_root = os.path.join(self.processed_dir, 'versions')
        os.makedirs(versions_root, exist_ok=True)

        if version_name is None:
            version_name = datetime.now().strftime('v_%Y%m%d_%H%M%S')

        version_dir = os.path.join(versions_root, version_name)
        if os.path.exists(version_dir):
            raise FileExistsError(f"version already exists: {version_dir}")

        shutil.copytree(source_dir, version_dir)
        version_meta = {
            'version': version_name,
            'source_dir': os.path.abspath(source_dir),
            'created_at': datetime.now().isoformat(),
            'metadata': metadata or {}
        }
        with open(os.path.join(version_dir, 'version_meta.json'), 'w', encoding='utf-8') as f:
            json.dump(version_meta, f, indent=2, ensure_ascii=False)

        return version_dir

    def list_dataset_versions(self) -> List[str]:
        """List all dataset version snapshots."""
        versions_root = os.path.join(self.processed_dir, 'versions')
        if not os.path.exists(versions_root):
            return []

        versions = [
            d for d in os.listdir(versions_root)
            if os.path.isdir(os.path.join(versions_root, d))
        ]
        return sorted(versions, reverse=True)


class CustomImageDataset(Dataset):
    """自定义图像数据集"""

    def __init__(self, image_paths: List[str], labels: List[int],
                 transform=None):
        """
        Args:
            image_paths: 图像路径列表
            labels: 标签列表
            transform: 图像变换
        """
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert('RGB')

        if self.transform:
            image = self.transform(image)

        label = self.labels[idx]
        return image, label
