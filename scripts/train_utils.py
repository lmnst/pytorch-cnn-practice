"""Shared training utilities for the sanity-check FashionMNIST runs.

The same data pipeline, train step, eval step, plotting, and metrics IO
are reused across every model in the repo. The intent is to keep each
per-model run small and to avoid duplicating the boilerplate that lived
inside each old per-folder script.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from torchvision.datasets import FashionMNIST

REPO_ROOT = Path(__file__).resolve().parent.parent
RUNS_DIR = REPO_ROOT / "runs"
DATA_DIR = REPO_ROOT / "data"
METRICS_PATH = RUNS_DIR / "metrics.json"


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def device_string() -> str:
    if torch.cuda.is_available():
        return f"cuda:0 ({torch.cuda.get_device_name(0)})"
    return "cpu"


def get_dataloaders(input_size: int, batch_size: int, num_workers: int = 0):
    """Return (train_loader, val_loader) for FashionMNIST resized to
    `input_size` on both spatial dims. The 60k training set is split 80/20
    with a fixed generator so re-runs use the same partition.

    num_workers defaults to 0 because PowerShell on Windows is the target
    environment and higher worker counts have been observed to cause
    spawn-related instability there.
    """
    transform = transforms.Compose(
        [
            transforms.Resize(input_size),
            transforms.ToTensor(),
        ]
    )
    full_train = FashionMNIST(
        root=str(DATA_DIR), train=True, transform=transform, download=True
    )
    train_len = round(0.8 * len(full_train))
    val_len = len(full_train) - train_len
    generator = torch.Generator().manual_seed(0)
    train_set, val_set = random_split(
        full_train, [train_len, val_len], generator=generator
    )

    pin = torch.cuda.is_available()
    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin,
    )
    return train_loader, val_loader


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def train_one_epoch(model, loader, optimizer, criterion, device):
    """Run one full training pass.

    Returns a tuple (average_loss, step_losses). The per-step loss list
    is kept so that callers can plot the within-epoch loss curve, which
    is more informative than a single epoch-averaged number when only
    one or two epochs are run.
    """
    model.train()
    running_loss = 0.0
    n = 0
    step_losses = []
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        bs = x.size(0)
        loss_val = loss.item()
        running_loss += loss_val * bs
        n += bs
        step_losses.append(loss_val)
    return running_loss / n, step_losses


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    n = 0
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        out = model(x)
        loss = criterion(out, y)
        running_loss += loss.item() * x.size(0)
        correct += (out.argmax(dim=1) == y).sum().item()
        n += x.size(0)
    return running_loss / n, correct / n


def save_loss_curve(model_display_name, step_losses, val_losses, val_accs, save_path: Path):
    """Plot the per-step training loss for the whole run and overlay the
    end-of-epoch validation numbers in a text box.

    Per-step granularity is used (rather than per-epoch averages) because
    with only one or two epochs the per-epoch series is too short to
    show the actual learning dynamic. The within-epoch curve is what
    surfaces things like batchnorm-driven smoothness or early-epoch
    noise.
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    num_epochs = len(val_losses)
    steps_per_epoch = len(step_losses) // num_epochs if num_epochs > 0 else 0

    fig, ax = plt.subplots(figsize=(8, 4.5))
    steps = list(range(1, len(step_losses) + 1))
    ax.plot(steps, step_losses, linewidth=1, color="steelblue")

    if num_epochs > 1:
        for i in range(1, num_epochs):
            ax.axvline(x=i * steps_per_epoch, color="gray", linestyle=":", alpha=0.5)

    val_text = "\n".join(
        f"epoch {i + 1}: val_loss={vl:.4f}  val_acc={va:.4f}"
        for i, (vl, va) in enumerate(zip(val_losses, val_accs))
    )
    ax.text(
        0.98,
        0.95,
        val_text,
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=9,
        family="monospace",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor="gray"),
    )

    ax.set_xlabel("training step")
    ax.set_ylabel("train loss")
    ax.set_title(f"{model_display_name} per-step training loss on FashionMNIST")

    fig.tight_layout()
    fig.savefig(save_path, dpi=110)
    plt.close(fig)


def append_run_record(record: dict, path: Path = METRICS_PATH):
    """Append (or replace) a single model's run record in metrics.json.

    If a record for the same model name already exists, it is overwritten
    so re-running one model does not leave stale rows. New models append.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        data = json.loads(path.read_text(encoding="utf-8"))
    else:
        data = {
            "dataset": "FashionMNIST",
            "device": device_string(),
            "torch_version": torch.__version__,
            "runs": [],
        }
    name = record["model"]
    data["runs"] = [r for r in data["runs"] if r.get("model") != name]
    data["runs"].append(record)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def now_iso() -> str:
    return datetime.now().strftime("%Y-%m-%dT%H:%M:%S")


def is_cuda_oom(err: BaseException) -> bool:
    """Detect CUDA out-of-memory errors across torch versions.

    Newer torch exposes `torch.cuda.OutOfMemoryError`; older ones raise a
    plain RuntimeError whose message contains "out of memory". On very
    tight VRAM the failure also surfaces as cuBLAS/cuDNN allocation
    failures with messages like CUBLAS_STATUS_ALLOC_FAILED, which are
    the same root cause and should trigger the same fallback.
    """
    oom_cls = getattr(torch.cuda, "OutOfMemoryError", None)
    if oom_cls is not None and isinstance(err, oom_cls):
        return True
    s = str(err).lower()
    return "out of memory" in s or "alloc_failed" in s
