# AUTOTRAINER
An Automatic DNN TrainingProblem Detection and Repair System


## How effective is AUTOTRAINER in detecting and fixing training problems?
As we shown in our paper, AUTOTRAINER can detect all five kinds of training problems in the input model with **100%** detection rate and **no** false positives in our experiment.
And in the repair module, it fixes 97.33% of the buggy models, increasing the accuracy by 47.08% on average.

## How efficient is AUTOTRAINER in detecting and fixing training problems?
As we shown in our paer, the AUTOTRAINER has little time overhead in detection and repair (the specific value is different in different datasets, the averaged value is round 1%.) Most of the extra time AUTOTRAINER spent is used to retrain the model. Additionally, AUTOTRAINER has little memory overhead because it reuses data which has been collected (e.g., loss value and accuracy in training).

## How to use AUTOTRAINER?
Here we prepare two simple cases in xx which is based on the Cirlce and Blob dataset on Sklearn. You can just enter the corresponding folder and run [`demo.py`](./AutoTrainer/demo_case/Gradient_Vanish_Case/demo.py) directly to see how AUTOTRANING solve the problems in these case. 
```
pip install -r requirements.txt # install the environment
cd AutoTrainer/demo_case/Gradient_Vanish_Case
#or use `cd AutoTrainer/demo_case/Oscillating_Loss_Case`
python demo.py
```

If you want to try on your own dataset, you should rewrite the `demo.py` and add your dateset in it. Additionally, you should write your training configuration in `.pkl` file. This file should include the dataset, batch size in training, the details of the optimizer, the loss function and the max training epoch. AUTOTRAINER has only on function `model_train()` to auto train your model. You can refer to the `demo.py` to use it.

## How can I set the configurable parameters?
Most of the configurable parameters can be set in `params` in the `demo.py` and the `model_train()`.

## Our Experiment Data
All Experiment Data(e.g., models, training configurations) can be found in https://drive.google.com/file/d/1AnzEwQZtKXAXA6jo4xGdhRLuAjnUFMLd/view?usp=sharing.

More information will be updated later
