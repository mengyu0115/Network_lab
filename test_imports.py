"""
Basic import smoke test.
"""
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

print("=" * 60)
print("Import smoke test")
print("=" * 60)

try:
    import torch  # noqa: F401
except Exception as exc:
    print(f"[SKIP] torch is unavailable in the current environment: {exc}")
    print("=" * 60)
    print("Import smoke test finished")
    print("=" * 60)
    raise SystemExit(0)

checks = [
    ("attacks", "from src.attacks import FGSM, PGD, CarliniWagner"),
    ("models", "from src.models import ModelLoader, load_model"),
    ("data_manager", "from src.data_manager import DatasetManager, CustomImageDataset"),
    ("evaluation", "from src.evaluation import AttackEvaluator, CLIPEvaluator, CLIPMultimodalEvaluator"),
    ("visualization", "from src.visualization import Visualizer"),
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
