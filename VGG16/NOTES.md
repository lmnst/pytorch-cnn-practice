# VGG16

VGG16 (Simonyan and Zisserman, 2014) on FashionMNIST resized to
224x224. Five blocks of 3x3 convs separated by max pool, then three
FC layers with dropout 0.5. 134.3M parameters.

Training did not complete on this hardware. At batch=16 the backward
pass failed with CUBLAS_STATUS_ALLOC_FAILED, which is cuBLAS
reporting an out-of-memory condition when setting up its handle. The
dynamic batch fallback halved to 8 and retried, but that attempt
then hit CUBLAS_STATUS_EXECUTION_FAILED. The second error is most
likely a follow-on from the first: an allocation failure can leave
the cuBLAS context in a state that torch.cuda.empty_cache() does not
fully reset, and subsequent kernel launches misbehave even when
nominal memory is available.

The architectural takeaway is parameter distribution. Of the 134.3M
total parameters, about 89% (~119.6M) sit in the three FC layers;
the convolutional stack itself is only ~14.7M. The first FC alone
(7x7x512 -> 4096) is 102.8M parameters. ResNet18 in this repo has
11.2M total, so VGG16 is 12x larger almost entirely because of the
FC head. Later architectures (Inception, ResNet) replaced that head
with global average pooling for exactly this reason: on a 2 GB
laptop GPU the VGG FC stack is what blows through the memory budget,
while ResNet18 fits comfortably at batch 64.

Takeaway: 134M parameters with about 89% in the FC head is the
actual reason VGG16 does not fit on a 2 GB GPU at usable batch
sizes. Replacing the FC head with global average pooling (as in
ResNet18 and GoogLeNet) is what makes 224x224 image models trainable
on consumer hardware.
