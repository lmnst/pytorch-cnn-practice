# LeNet

LeNet-5 (LeCun et al., 1998) on FashionMNIST. Sigmoid activations and
average pooling, matching the original paper.

1 epoch, batch 128, Adam lr=1e-3, on MX350: val_acc=0.6158,
train_loss_avg=1.65, val_loss=0.98. The per-step loss sits at the
random-guess baseline (around 2.30, the cross-entropy of a 10-way
uniform distribution) until step ~200 of 375, then drops to ~1.0 by
the end. Sigmoid saturation explains the long flat start. For
contrast, AlexNet with ReLU under the same budget is already at
train_loss 0.54 by a quarter of its own epoch.

The epoch-averaged train_loss (1.65) is higher than val_loss (0.98).
This is not under-training. train_loss is averaged over all the
epoch's steps including the early bad ones; val_loss is computed on
the model state after the epoch finished. For the first epoch the
two numbers are not directly comparable.

Takeaway: 61k params reach val_acc 0.62 in one epoch on FashionMNIST,
but the sigmoid path wastes the first half of the epoch in
saturated-gradient stall.
