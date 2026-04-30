"""
Train image classifiers used by the attack demo.

Examples:
    python scripts/train_classifier.py --dataset cifar10 --model resnet18 --epochs 5
    python scripts/train_classifier.py --dataset mnist --model resnet18 --epochs 3
"""
import argparse
import os
import sys
from typing import Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.models import ModelLoader, load_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a CIFAR-10/MNIST classifier for image attacks.")
    parser.add_argument("--dataset", choices=["cifar10", "cifar-10", "mnist"], required=True)
    parser.add_argument("--model", choices=["resnet18", "resnet50"], default="resnet18")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--freeze-backbone", action="store_true", help="Only train the final classifier head.")
    parser.add_argument("--max-train-samples", type=int, default=0, help="Optional quick-test subset size.")
    parser.add_argument("--max-test-samples", type=int, default=0, help="Optional quick-test subset size.")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def normalize_dataset_name(name: str) -> str:
    return "cifar-10" if name.lower() in {"cifar10", "cifar-10"} else "mnist"


def build_transforms(dataset_name: str, image_size: int) -> Tuple[transforms.Compose, transforms.Compose]:
    common = [transforms.Resize((image_size, image_size))]
    if dataset_name == "mnist":
        common.append(transforms.Grayscale(num_output_channels=3))

    train_transform = transforms.Compose(
        common
        + ([transforms.RandomHorizontalFlip()] if dataset_name == "cifar-10" else [])
        + [transforms.ToTensor()]
    )
    test_transform = transforms.Compose(common + [transforms.ToTensor()])
    return train_transform, test_transform


def build_datasets(dataset_name: str, image_size: int):
    data_root = os.path.join(PROJECT_ROOT, "data", "raw")
    train_transform, test_transform = build_transforms(dataset_name, image_size)

    if dataset_name == "cifar-10":
        train_set = datasets.CIFAR10(data_root, train=True, download=True, transform=train_transform)
        test_set = datasets.CIFAR10(data_root, train=False, download=True, transform=test_transform)
    else:
        train_set = datasets.MNIST(data_root, train=True, download=True, transform=train_transform)
        test_set = datasets.MNIST(data_root, train=False, download=True, transform=test_transform)

    return train_set, test_set


def maybe_subset(dataset, max_samples: int):
    if max_samples and max_samples > 0:
        return Subset(dataset, range(min(max_samples, len(dataset))))
    return dataset


def freeze_backbone(model: nn.Module) -> None:
    for param in model.parameters():
        param.requires_grad = False

    if hasattr(model, "fc"):
        for param in model.fc.parameters():
            param.requires_grad = True
    elif hasattr(model, "classifier"):
        for param in model.classifier.parameters():
            param.requires_grad = True
    else:
        raise ValueError("Unsupported model head for freeze-backbone mode.")


def run_epoch(model, loader, criterion, device, optimizer=None):
    is_train = optimizer is not None
    model.train(is_train)

    total_loss = 0.0
    total_correct = 0
    total_count = 0

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        if is_train:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(is_train):
            outputs = model(images)
            loss = criterion(outputs, labels)
            if is_train:
                loss.backward()
                optimizer.step()

        total_loss += float(loss.item()) * images.size(0)
        total_correct += int((outputs.argmax(dim=1) == labels).sum().item())
        total_count += int(images.size(0))

    avg_loss = total_loss / max(total_count, 1)
    accuracy = 100.0 * total_correct / max(total_count, 1)
    return avg_loss, accuracy


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)

    dataset_name = normalize_dataset_name(args.dataset)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    num_classes = 10

    train_set, test_set = build_datasets(dataset_name, args.image_size)
    train_set = maybe_subset(train_set, args.max_train_samples)
    test_set = maybe_subset(test_set, args.max_test_samples)

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device == "cuda"),
    )
    test_loader = DataLoader(
        test_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device == "cuda"),
    )

    model = load_model(
        args.model,
        pretrained=True,
        num_classes=num_classes,
        device=device,
        dataset_name=dataset_name,
    )
    if args.freeze_backbone:
        freeze_backbone(model)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    checkpoint_path = ModelLoader.get_finetune_checkpoint_path(args.model, dataset_name)
    if checkpoint_path is None:
        raise ValueError(f"No checkpoint path mapping for model={args.model}, dataset={dataset_name}")
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)

    best_acc = -1.0
    print(f"device={device}, model={args.model}, dataset={dataset_name}, checkpoint={checkpoint_path}")
    print(f"train_samples={len(train_set)}, test_samples={len(test_set)}, freeze_backbone={args.freeze_backbone}")

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = run_epoch(model, train_loader, criterion, device, optimizer)
        test_loss, test_acc = run_epoch(model, test_loader, criterion, device)

        print(
            f"epoch {epoch:02d}/{args.epochs} "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.2f}% "
            f"test_loss={test_loss:.4f} test_acc={test_acc:.2f}%"
        )

        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "model_name": args.model,
                    "dataset_name": dataset_name,
                    "num_classes": num_classes,
                    "image_size": args.image_size,
                    "best_test_accuracy": best_acc,
                    "epoch": epoch,
                },
                checkpoint_path,
            )
            print(f"saved best checkpoint: {checkpoint_path}")

    print(f"done. best_test_accuracy={best_acc:.2f}%")


if __name__ == "__main__":
    main()
