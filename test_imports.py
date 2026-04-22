"""
Basic import smoke test.
"""
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

print("=" * 60)
print("Import smoke test")
print("=" * 60)

checks = [
    ("attacks.base", "from src.attacks import base"),
    ("attacks.fgsm", "from src.attacks import fgsm"),
    ("attacks.pgd", "from src.attacks import pgd"),
    ("attacks.cw", "from src.attacks import cw"),
    ("attacks.multimodal_clip", "from src.attacks import multimodal_clip"),
    ("models.model_loader", "from src.models import model_loader"),
    ("data_manager.dataset_manager", "from src.data_manager import dataset_manager"),
    ("evaluation.metrics", "from src.evaluation import metrics"),
    ("evaluation.multimodal_metrics", "from src.evaluation import multimodal_metrics"),
    ("visualization.visualizer", "from src.visualization import visualizer"),
]

for name, stmt in checks:
    try:
        exec(stmt, {})
        print(f"[PASS] {name}")
    except Exception as e:
        print(f"[FAIL] {name}: {e}")

print("=" * 60)
print("Import smoke test finished")
print("=" * 60)
