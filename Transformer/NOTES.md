# ViT-tiny

A minimal Vision Transformer for FashionMNIST: 7x7 patches on 28x28
input (16 tokens), embed_dim=64, 4 encoder blocks, 4 heads, learnable
CLS token and positional embedding, pre-norm attention + MLP per
block, classify from the CLS token. 205K params.

1 epoch, batch 128, Adam lr=1e-3, on MX350: val_acc=0.7963,
train_loss_avg=0.91, val_loss=0.57, 31.9 s.

The result was the opposite of the prediction (that ViT should
underperform a CNN at one epoch because it lacks conv locality bias).
On the same 28x28 input as LeNet, ViT-tiny reached val_acc 0.7963 vs
LeNet's 0.6158 with 3.3x the parameters (205K vs 61K). 7x7 patches on
a 28x28 image leave only 16 tokens, so the conv locality bias ViT
lacks is not particularly useful here. ViT-tiny's LayerNorm, GELU,
and residual block design also let it leave the random-guess regime
inside the first quarter of the epoch, where LeNet's sigmoid path
was still stalled.

This is a small-input effect, not a general claim. At 224x224 the
deeper CNNs in this repo still win (AlexNet 0.8439, GoogLeNet 0.8565).

Takeaway: at 28x28 a tiny ViT beats LeNet by 18 percentage points in
one epoch on FashionMNIST. The CNN inductive bias matters less the
smaller the input.
