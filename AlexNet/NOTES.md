# AlexNet

AlexNet (Krizhevsky et al., 2012) on FashionMNIST resized to 227x227.
ReLU activations, max pooling, two FC layers with dropout 0.5.

1 epoch, batch 64, Adam lr=1e-3, on MX350: val_acc=0.8439,
train_loss_avg=0.63, val_loss=0.43. Training time was 352 s, the
slowest non-VGG model because the 4096-unit FC head dominates compute
on 227x227 inputs.

The ReLU vs LeNet's sigmoid comparison is the headline. Both start at
the random-guess CE of about 2.30. After 10% of the epoch (step 75
of 750), AlexNet's train loss is already at 1.08; LeNet at the same
fraction of its own epoch is still at 2.31, essentially unchanged.
AlexNet reaches 0.54 by 25%; LeNet stays at 2.31 until step 200 of
375. The same architecture family with sigmoid would spend most of
the first epoch in saturated-gradient stall.

Grouped convolutions from the original paper were omitted. On
FashionMNIST that historical detail would add complexity without
buying anything.

Takeaway: ReLU lets AlexNet leave random-guess within the first 10%
of the epoch and reach val_acc 0.84 in a single pass on FashionMNIST.
