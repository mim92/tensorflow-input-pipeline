# tensorflow input-pipeline example
This is tensorflow input-pipeline example code.

## Requirements
- Python 3.6.4
- tensorflow 1.9.0

## Training
```
$ python train.py --n_thread 1
Epoch 1/10
100%|█████████████████████████| 1875/1875 [01:43<00:00, 18.13it/s, train_acc=0.687, train_loss=31.8]
train/acc: 0.6868, train/loss: 31.8274
valid/acc: 0.8502, valid/loss: 14.6315
.
.
.
$ python train.py --n_thread 4
Epoch 1/10
100%|█████████████████████████| 1875/1875 [00:26<00:00, 69.64it/s, train_acc=0.668, train_loss=33.4]
train/acc: 0.6685, train/loss: 33.4167
valid/acc: 0.8849, valid/loss: 12.0425
.
.
.

```