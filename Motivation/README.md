# Motivation

## Usage

To reproduce the two cases, please run the following codes.

```bash
$ cd ./DyingReLU
# or use `cd ./OscillatingLoss`
$ python train.py
```

1. The [test case](./DyingReLU/train.py) in 'DyingReLU' direction shows that training problem occurrence is highly random. In our previous test, DyingReLU problem occurs in **50%** of the cases, and **50%** of the cases will behave normally and achieve high training accuracy.
2. The [test case](./OscillatingLoss/train.py) in 'OscillatingLoss' direction shows that the time when a training problem occurs is random. In our previous test, Oscillating Loss problem occurs in the first **10** training epoch in **50%** of the cases. In other cases, this problem happens in later epochs and **29%** cases even don't perform this problem in the training.
   
![avatar](https://github.com/shiningrain/tmpfigure/blob/master/TDSC/Figure1.png)