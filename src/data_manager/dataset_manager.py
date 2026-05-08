"""
Data Manager Module
数据集管理模块
"""
import os
import json
import re
import torch
import numpy as np
from PIL import Image
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets
from typing import List, Tuple, Optional, Dict, Any
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
            max_val = float(perturbation.max().item())
            if max_val > 1e-12:
                perturbation = torch.clamp(perturbation / max_val, 0.0, 1.0)
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

    def _to_serializable(self, value: Any) -> Any:
        """Convert nested values to JSON-serializable types."""
        if isinstance(value, torch.Tensor):
            if value.numel() == 1:
                return float(value.item())
            return value.detach().cpu().tolist()
        if isinstance(value, np.ndarray):
            if value.size == 1:
                return float(value.item())
            return value.tolist()
        if isinstance(value, (np.floating,)):
            return float(value)
        if isinstance(value, (np.integer,)):
            return int(value)
        if isinstance(value, dict):
            return {str(k): self._to_serializable(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [self._to_serializable(v) for v in value]
        return value

    def save_experiment_record(self,
                               experiment_name: str,
                               metadata: Dict,
                               original_image: Optional[torch.Tensor] = None,
                               adversarial_image: Optional[torch.Tensor] = None) -> str:
        """
        Save a generic experiment record (supports text/multimodal).

        If original/adversarial images are provided, also store
        original_0/adversarial_0/perturbation_0 for history preview.
        """
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        save_dir = os.path.join(self.adversarial_dir, f"{experiment_name}_{timestamp}")
        os.makedirs(save_dir, exist_ok=True)

        if original_image is not None and adversarial_image is not None:
            original_image = original_image.detach().cpu()
            adversarial_image = adversarial_image.detach().cpu()
            if original_image.shape[-2:] != adversarial_image.shape[-2:]:
                original_image = F.interpolate(
                    original_image.unsqueeze(0),
                    size=adversarial_image.shape[-2:],
                    mode='bilinear',
                    align_corners=False,
                ).squeeze(0)
            self._save_image(original_image, os.path.join(save_dir, 'original_0.png'))
            self._save_image(adversarial_image, os.path.join(save_dir, 'adversarial_0.png'))

            perturbation = (adversarial_image - original_image).abs()
            max_val = float(perturbation.max().item())
            if max_val > 1e-12:
                perturbation = torch.clamp(perturbation / max_val, 0.0, 1.0)
            self._save_image(perturbation, os.path.join(save_dir, 'perturbation_0.png'))

        payload = {
            'attack_name': experiment_name,
            'timestamp': timestamp,
            **self._to_serializable(metadata)
        }
        if 'num_samples' not in payload:
            payload['num_samples'] = 1

        metadata_path = os.path.join(save_dir, 'metadata.json')
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)

        return save_dir

    def _deep_merge_dict(self, base: Dict[str, Any], updates: Dict[str, Any]) -> Dict[str, Any]:
        """Recursively merge nested metadata dictionaries."""
        for key, value in updates.items():
            if isinstance(base.get(key), dict) and isinstance(value, dict):
                base[key] = self._deep_merge_dict(base[key], value)
            else:
                base[key] = value
        return base

    def update_experiment_metadata(
        self,
        experiment_dir: str,
        updates: Dict[str, Any],
        merge: bool = True,
    ) -> str:
        """Update metadata.json for an existing experiment."""
        metadata_path = os.path.join(experiment_dir, 'metadata.json')
        if not os.path.exists(metadata_path):
            raise FileNotFoundError(f"metadata.json not found: {metadata_path}")

        payload = {}
        if merge:
            with open(metadata_path, 'r', encoding='utf-8') as f:
                payload = json.load(f)
            payload = self._deep_merge_dict(payload, self._to_serializable(updates))
        else:
            payload = self._to_serializable(updates)

        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)

        return metadata_path

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

    def _safe_name(self, value: str) -> str:
        value = re.sub(r'[^0-9a-zA-Z\u4e00-\u9fa5._-]+', '_', value.strip())
        return value.strip('_') or 'sample'

    def save_uploaded_samples(
        self,
        uploaded_files,
        annotations: Optional[List[Dict[str, Any]]] = None,
        collection_name: Optional[str] = None,
    ) -> str:
        """Save uploaded image samples and generate a manifest.json."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        collection_name = self._safe_name(collection_name or f'uploads_{timestamp}')
        save_dir = os.path.join(self.raw_dir, 'uploads', f'{collection_name}_{timestamp}')
        os.makedirs(save_dir, exist_ok=True)

        manifest = []
        for idx, file_obj in enumerate(uploaded_files):
            annotation = annotations[idx] if annotations and idx < len(annotations) else {}
            original_name = getattr(file_obj, 'name', f'sample_{idx}.png')
            stem = self._safe_name(os.path.splitext(original_name)[0])
            suffix = os.path.splitext(original_name)[1].lower() or '.png'
            if hasattr(file_obj, 'seek'):
                try:
                    file_obj.seek(0)
                except Exception:
                    pass
            image = Image.open(file_obj).convert('RGB')
            file_name = f'{idx:03d}_{stem}{suffix if suffix in {".png", ".jpg", ".jpeg"} else ".png"}'
            file_path = os.path.join(save_dir, file_name)
            image.save(file_path)

            record = {
                'index': idx,
                'original_name': original_name,
                'file_name': file_name,
                'file_path': file_path,
                'label': annotation.get('label'),
                'split': annotation.get('split', 'unspecified'),
                'notes': annotation.get('notes', ''),
                'tags': annotation.get('tags', []),
                'created_at': timestamp,
            }
            manifest.append(record)

        manifest_payload = {
            'collection_name': collection_name,
            'timestamp': timestamp,
            'num_samples': len(manifest),
            'samples': manifest,
        }
        with open(os.path.join(save_dir, 'manifest.json'), 'w', encoding='utf-8') as f:
            json.dump(manifest_payload, f, indent=2, ensure_ascii=False)

        return save_dir

    def update_uploaded_collection_manifest(
        self,
        collection_dir: str,
        samples: List[Dict[str, Any]],
    ) -> str:
        """Update an uploaded collection manifest with edited annotations."""
        manifest = self.load_uploaded_collection(collection_dir)
        samples_by_name = {
            item.get('file_name'): item
            for item in manifest.get('samples', [])
        }

        for sample in samples:
            file_name = sample.get('file_name')
            if file_name not in samples_by_name:
                continue
            target = samples_by_name[file_name]
            target['label'] = sample.get('label', target.get('label'))
            target['split'] = sample.get('split', target.get('split', 'unspecified'))
            target['notes'] = sample.get('notes', target.get('notes', ''))
            tags = sample.get('tags', target.get('tags', []))
            if isinstance(tags, str):
                tags = [tag.strip() for tag in tags.split(',') if tag.strip()]
            target['tags'] = tags

        manifest['samples'] = list(samples_by_name.values())
        manifest['num_samples'] = len(manifest['samples'])
        manifest['updated_at'] = datetime.now().strftime('%Y%m%d_%H%M%S')

        with open(os.path.join(collection_dir, 'manifest.json'), 'w', encoding='utf-8') as f:
            json.dump(manifest, f, indent=2, ensure_ascii=False)

        return collection_dir

    def export_uploaded_collection_manifest(self, collection_dir: str, export_format: str = 'json') -> bytes:
        """Export an uploaded collection manifest as CSV or JSON bytes."""
        manifest = self.load_uploaded_collection(collection_dir)
        samples = manifest.get('samples', [])

        if export_format == 'json':
            return json.dumps(manifest, indent=2, ensure_ascii=False).encode('utf-8')

        try:
            import pandas as pd
        except ImportError as exc:
            raise ImportError("pandas is required for CSV export") from exc

        return pd.DataFrame(samples).to_csv(index=False).encode('utf-8')

    def list_uploaded_collections(self) -> List[str]:
        """List uploaded sample collections stored under raw/uploads."""
        uploads_root = os.path.join(self.raw_dir, 'uploads')
        if not os.path.exists(uploads_root):
            return []

        collections = [
            d for d in os.listdir(uploads_root)
            if os.path.isdir(os.path.join(uploads_root, d))
        ]
        return sorted(collections, reverse=True)

    def load_uploaded_collection(self, collection_dir: str) -> Dict[str, Any]:
        """Load uploaded sample manifest."""
        manifest_path = os.path.join(collection_dir, 'manifest.json')
        if not os.path.exists(manifest_path):
            raise FileNotFoundError(f'manifest.json not found: {manifest_path}')
        with open(manifest_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def _flatten_metric_items(self, metrics: Dict[str, Any], prefix: str = '') -> Dict[str, Any]:
        flattened = {}
        for key, value in metrics.items():
            flat_key = f'{prefix}{key}' if prefix else str(key)
            if isinstance(value, dict):
                flattened.update(self._flatten_metric_items(value, prefix=f'{flat_key}.'))
            else:
                flattened[flat_key] = self._to_serializable(value)
        return flattened

    def build_experiment_index(self) -> List[Dict[str, Any]]:
        """Build a flat summary for all stored adversarial experiments."""
        rows = []
        for exp_name in self.list_experiments():
            exp_path = os.path.join(self.adversarial_dir, exp_name)
            try:
                metadata = self.load_adversarial_samples(exp_path)
            except Exception:
                continue

            row = {
                'name': exp_name,
                'path': exp_path,
                'attack': metadata.get('attack_name', metadata.get('attack', '-')),
                'model': metadata.get('model', '-'),
                'dataset': metadata.get('dataset', '-'),
                'task_type': metadata.get('task_type', 'classification'),
                'target_family': metadata.get('target_family', ''),
                'timestamp': metadata.get('timestamp', ''),
                'num_samples': int(metadata.get('num_samples', 0) or 0),
            }

            metrics = metadata.get('metrics', {})
            if isinstance(metrics, dict):
                row.update(self._flatten_metric_items(metrics, prefix='metric.'))

            multimodal = metadata.get('multimodal_evaluations', {})
            if isinstance(multimodal, dict):
                for family, family_metrics in multimodal.items():
                    if isinstance(family_metrics, dict):
                        row.update(self._flatten_metric_items(family_metrics, prefix=f'{family}.'))

            rows.append(row)

        return rows

    def export_experiment_index(self, export_format: str = 'csv') -> bytes:
        """Export experiment index as CSV or JSON bytes."""
        rows = self.build_experiment_index()
        if export_format == 'json':
            return json.dumps(rows, indent=2, ensure_ascii=False).encode('utf-8')

        try:
            import pandas as pd
        except ImportError as exc:
            raise ImportError("pandas is required for CSV export") from exc

        df = pd.DataFrame(rows)
        return df.to_csv(index=False).encode('utf-8')

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
