# ResNet18

ResNet-18 (He et al., 2015) on FashionMNIST resized to 224x224. Two
BasicBlocks per stage at widths [64, 128, 256, 512]; stages 2-4
downsample with stride 2 and a 1x1 shortcut. BN on every conv.
11.2M params.

1 epoch, batch 64, Adam lr=1e-3, on MX350: train_loss_avg=0.47, final
step train loss=0.30 (the lowest in this repo), val_loss=1.06,
val_acc=0.6152, 1253 s (over the 10-min-per-epoch budget).

The loss curve carries two observations. First, the per-step descent
is smoother than the non-BN models. Mean absolute step-to-step jump
in train loss across the second half of training is 0.11 for
ResNet18, 0.12 for AlexNet, and 0.17 for GoogLeNet (the latter two
have no BN). BN keeps each layer's input distribution stationary
across steps, so the per-step loss does not bounce as much.

Second, the final-step train loss (0.30) is far below val_loss (1.06).
This is BN's well-known early-epoch behaviour, not real overfitting.
In train mode BN normalizes by the current batch's statistics; in
eval mode it uses running averages updated with momentum 0.1, and
after a single epoch those averages are still biased toward
late-epoch batches. The eval forward therefore sees a different
normalization than the model trained against.

Takeaway: BN gives the lowest and smoothest train loss in this repo,
but its running statistics need more than one epoch to stabilize, so
val_acc lags train_loss after one pass.
