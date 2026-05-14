# GoogLeNet

GoogLeNet (Szegedy et al., 2014) on FashionMNIST resized to 224x224.
Stacked Inception blocks with four parallel paths (1x1 conv, 1x1+3x3,
1x1+5x5, maxpool+1x1) concatenated on the channel axis, plus a
global average pool head.

1 epoch on MX350: requested batch 64 but the dynamic fallback halved
it to 32 after a CUDA OOM (224x224 at batch 64 did not fit on 2 GB
VRAM). Adam lr=1e-3: val_acc=0.8565, train_loss_avg=0.72,
val_loss=0.38, 867 s. The run exceeded the 10-min-per-epoch budget;
kept because it completed and the per-step curve is informative.

Parameter efficiency is the headline. The 1x1 convs at the front of
each Inception branch act as channel bottlenecks, so the internal
widths stay narrow. Total: 5.9M parameters, 23x fewer than VGG16
(134.3M) and 10x fewer than AlexNet (58.3M). Despite the much smaller
count, val_acc beats AlexNet (0.8565 vs 0.8439) under the same
1-epoch budget on FashionMNIST.

Takeaway: Inception's 1x1 bottlenecks let GoogLeNet outperform
AlexNet's val accuracy with 10x fewer parameters in one epoch.
