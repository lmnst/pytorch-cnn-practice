"""Run a sanity-check training pass on one of the implemented models.

Each model is trained on FashionMNIST for a small number of epochs. The
point is to exercise the training loop end to end and produce a loss
curve, not to chase accuracy. Per-model defaults (input size, batch
size) live in MODEL_REGISTRY below.

Examples:
    python scripts/train.py --model lenet
    python scripts/train.py --model lenet --epochs 2
    python scripts/train.py --model vgg16 --batch-size 8
    python scripts/train.py --model all
"""

from __future__ import annotations

import argparse
import gc
import sys
import time
import traceback
from pathlib import Path

import torch
from torch import nn

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.train_utils import (
    RUNS_DIR,
    append_run_record,
    count_parameters,
    device_string,
    evaluate,
    get_dataloaders,
    get_device,
    is_cuda_oom,
    now_iso,
    save_loss_curve,
    train_one_epoch,
)


def build_lenet():
    from LeNet.model import LeNet

    return LeNet()


def build_alexnet():
    from AlexNet.model import AlexNet

    return AlexNet()


def build_vgg16():
    from VGG16.model import VGG16

    return VGG16()


def build_googlenet():
    from GoogLeNet.model import GoogLeNet, Inception

    return GoogLeNet(Inception)


def build_resnet18():
    from ResNet18.model import ResNet18

    return ResNet18()


def build_vit_tiny():
    from Transformer.model import ViT_Tiny

    return ViT_Tiny()


MODEL_REGISTRY = {
    "lenet": dict(build=build_lenet, input_size=28, default_batch=128, display="LeNet", weight_dir="LeNet"),
    "alexnet": dict(build=build_alexnet, input_size=227, default_batch=64, display="AlexNet", weight_dir="AlexNet"),
    "vgg16": dict(build=build_vgg16, input_size=224, default_batch=16, display="VGG16", weight_dir="VGG16"),
    "googlenet": dict(build=build_googlenet, input_size=224, default_batch=64, display="GoogLeNet", weight_dir="GoogLeNet"),
    "resnet18": dict(build=build_resnet18, input_size=224, default_batch=64, display="ResNet18", weight_dir="ResNet18"),
    "vit_tiny": dict(build=build_vit_tiny, input_size=28, default_batch=128, display="ViT-tiny", weight_dir="Transformer"),
}


def train_one_model(name: str, epochs: int, batch_size: int | None, min_batch: int = 4) -> None:
    """Train one model with dynamic OOM fallback.

    If a CUDA out-of-memory error is hit, the model, optimizer, and
    dataloaders are torn down, the cache is emptied, and the loop retries
    at half the batch size. The loop gives up once batch size would drop
    below `min_batch`, records `status="oom"` in metrics.json, and moves on.
    """
    spec = MODEL_REGISTRY[name]
    input_size = spec["input_size"]
    batch = batch_size if batch_size is not None else spec["default_batch"]
    device = get_device()
    criterion = nn.CrossEntropyLoss()

    print(f"\n=== Training {spec['display']} ===")
    print(
        f"device={device_string()}  input_size={input_size}x{input_size}  "
        f"target_batch_size={batch}  epochs={epochs}"
    )

    while True:
        model = None
        try:
            model = spec["build"]().to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
            train_loader, val_loader = get_dataloaders(input_size, batch)
            params = count_parameters(model)
            print(f"params={params:,}")

            train_losses_avg, all_step_losses, val_losses, val_accs = [], [], [], []
            start = time.time()
            for epoch in range(1, epochs + 1):
                t0 = time.time()
                train_loss_avg, step_losses = train_one_epoch(
                    model, train_loader, optimizer, criterion, device
                )
                val_loss, val_acc = evaluate(model, val_loader, criterion, device)
                train_losses_avg.append(train_loss_avg)
                all_step_losses.extend(step_losses)
                val_losses.append(val_loss)
                val_accs.append(val_acc)
                print(
                    f"epoch {epoch}/{epochs}  "
                    f"train_loss={train_loss_avg:.4f}  val_loss={val_loss:.4f}  "
                    f"val_acc={val_acc:.4f}  ({time.time() - t0:.1f}s)"
                )
            total = time.time() - start

            curve_path = RUNS_DIR / f"{name}_loss.png"
            save_loss_curve(spec["display"], all_step_losses, val_losses, val_accs, curve_path)

            weight_dir = REPO_ROOT / spec["weight_dir"]
            weight_dir.mkdir(exist_ok=True)
            torch.save(model.state_dict(), weight_dir / "best_model.pth")

            append_run_record(
                {
                    "model": name,
                    "display_name": spec["display"],
                    "status": "completed",
                    "skip_reason": None,
                    "params": params,
                    "input_size": [1, input_size, input_size],
                    "batch_size": batch,
                    "epochs": epochs,
                    "train_loss_per_epoch": train_losses_avg,
                    "train_loss_per_step": all_step_losses,
                    "val_loss_per_epoch": val_losses,
                    "val_acc_per_epoch": val_accs,
                    "best_val_acc": max(val_accs),
                    "training_time_seconds": round(total, 1),
                    "loss_curve_path": f"runs/{name}_loss.png",
                    "timestamp": now_iso(),
                }
            )
            print(f"done {spec['display']} in {total:.1f}s, curve at {curve_path}")
            return

        except RuntimeError as e:
            if is_cuda_oom(e):
                if model is not None:
                    del model
                torch.cuda.empty_cache()
                gc.collect()
                if batch > min_batch:
                    new_batch = max(min_batch, batch // 2)
                    print(f"OOM at batch={batch}, retrying at batch={new_batch}")
                    batch = new_batch
                    continue
                print(f"OOM at batch={batch}, giving up on {name}")
                append_run_record(
                    {
                        "model": name,
                        "display_name": spec["display"],
                        "status": "oom",
                        "skip_reason": f"CUDA out of memory at batch_size={batch}",
                        "params": None,
                        "input_size": [1, input_size, input_size],
                        "batch_size": batch,
                        "epochs": epochs,
                        "train_loss_per_epoch": [],
                        "train_loss_per_step": [],
                        "val_loss_per_epoch": [],
                        "val_acc_per_epoch": [],
                        "best_val_acc": None,
                        "training_time_seconds": None,
                        "loss_curve_path": None,
                        "timestamp": now_iso(),
                    }
                )
                return
            raise


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--model",
        required=True,
        choices=list(MODEL_REGISTRY.keys()) + ["all"],
        help="model name, or 'all' to train every model in sequence",
    )
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="override the per-model default; useful when probing OOM behaviour",
    )
    return p.parse_args()


def main():
    args = parse_args()
    names = list(MODEL_REGISTRY.keys()) if args.model == "all" else [args.model]
    for name in names:
        try:
            train_one_model(name, args.epochs, args.batch_size)
        except Exception as e:
            print(f"!! {name} failed: {e}")
            traceback.print_exc()
            spec = MODEL_REGISTRY[name]
            append_run_record(
                {
                    "model": name,
                    "display_name": spec["display"],
                    "status": "failed",
                    "skip_reason": str(e),
                    "params": None,
                    "input_size": [1, spec["input_size"], spec["input_size"]],
                    "batch_size": args.batch_size or spec["default_batch"],
                    "epochs": args.epochs,
                    "train_loss_per_epoch": [],
                    "train_loss_per_step": [],
                    "val_loss_per_epoch": [],
                    "val_acc_per_epoch": [],
                    "best_val_acc": None,
                    "training_time_seconds": None,
                    "loss_curve_path": None,
                    "timestamp": now_iso(),
                }
            )


if __name__ == "__main__":
    main()
