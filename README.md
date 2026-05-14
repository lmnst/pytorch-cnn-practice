# pytorch-cnn-practice

Self-study repo. Line-by-line reimplementations of classic CNN
architectures (and a small Vision Transformer) for understanding, not
for state-of-the-art accuracy. Each model is trained briefly on
FashionMNIST as a sanity check that the training loop and the
architecture are wired up correctly.

## Sanity-check training results

Dataset: FashionMNIST (1 channel, 10 classes).
Hardware: NVIDIA GeForce MX350 (2 GB VRAM).
Stack: Python 3.9, PyTorch 2.3.0 with CUDA 11.8.
Training: `scripts/train.py`, 1 epoch, Adam lr=1e-3, CrossEntropyLoss.

| model     | params      | input size | val acc | training time |
|-----------|-------------|------------|---------|---------------|
| LeNet     | 61,706      | 28x28      | 0.6158  | 26 s          |
| AlexNet   | 58,299,082  | 227x227    | 0.8439  | 352 s         |
| VGG16     | 134,300,362 | 224x224    | OOM     | n/a           |
| GoogLeNet | 5,927,850   | 224x224    | 0.8565  | 867 s         |
| ResNet18  | 11,178,378  | 224x224    | 0.6152  | 1253 s        |
| ViT-tiny  | 205,066     | 28x28      | 0.7963  | 32 s          |

Loss curves: `runs/<model>_loss.png`. Raw per-step training loss and
end-of-epoch validation numbers are in `runs/metrics.json`.

Trained 5 of 6 models. VGG16 did not fit on this 2 GB GPU even after
the OOM fallback halved its batch size from 16 to 8; see
`VGG16/NOTES.md` for the cuBLAS failure trace and the
param-distribution explanation. GoogLeNet (14.5 min) and ResNet18
(20.9 min) exceeded the 10-min-per-epoch budget set for this study
but were kept because they completed and their per-step loss curves
carry useful information.

A note on the ResNet18 row. val_acc 0.6152 looks bad on its own but
is consistent with BN running statistics not being stable after a
single epoch (the final-step train loss is 0.30, the lowest in this
repo). See `ResNet18/NOTES.md` for the full explanation.

## Models

Each folder contains a `model.py` and a `NOTES.md` recording what was
observed during training.

- `LeNet/` original LeCun et al. (1998) net with sigmoid and average pool
- `AlexNet/` 2012 ReLU + dropout, the network that started the deep era
- `VGG16/` deep 3x3 conv stack with a large FC head (the part that does not fit here)
- `GoogLeNet/` Inception blocks with 1x1 channel bottlenecks
- `ResNet18/` basic blocks with residual connections and batch norm
- `Transformer/` small Vision Transformer (`ViT_Tiny` class)

## Layout

    scripts/
        train.py            single entrypoint, dispatches by --model
        train_utils.py      dataloaders, train/eval steps, plotting, metrics IO
    runs/
        metrics.json        per-model run records, including per-step train loss
        <model>_loss.png    per-step train loss with val numbers annotated

## Run a single model

    python scripts/train.py --model lenet
    python scripts/train.py --model alexnet --epochs 2
    python scripts/train.py --model vgg16 --batch-size 8

The training loop halves the batch size and retries on a CUDA OOM, so
starting at a higher batch size and letting the script shrink it is
safe.
