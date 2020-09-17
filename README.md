# AUTOTRAINER

## TL;DR

Automatic detecting and fixing DNN training problem at run time.

## Repo Structure

1. AutoTrainer: It mainly contains the source codes of the AUTOTRAINER (the folder `data` and `utils`), two demo cases. You can find a easy start [here](./AutoTrainer/README.md). The way to run the demo cases has been shown [here](#Demo).
2. Motivation: It contains two test cases shown that 1) Training problem occurrence is highly random. 2) The time when a training problem occurs is random. The way to reproduce these cases can be found [here](./Motivation/README.md).
3. misc.: The [`README.md`](.README.md) shows how to use the our demos, the repo structure, the way to reproduce our experiments and our experiment results. And the `requirement.text` shows all the dependencies of AUTOTRAINER.

```
- AutoTrainer/                 
    - data/    
    - demo_case/  
        - Gradient_Vanish_Case/
        - Oscillating_Loss_Case/
    - utils/         
    - reproduce.py             
    - README.md                  
- Motivation/                      
    - DyingReLU/
    - OscillatingLoss/
    - README.md
- README.md
- requirements.txt
```


## Demo

Here we prepare [two simple cases](./AutoTrainer/demo_case) which is based on the Cirlce and Blob dataset on Sklearn. You can just enter the corresponding folder and run [`demo.py`](./AutoTrainer/demo_case/Gradient_Vanish_Case/demo.py) directly to see how AUTOTRAINER solve the problems in these case.

```bash
$ pip install -r requirements.txt
$ cd AutoTrainer/demo_case/Gradient_Vanish_Case
# or use `cd AutoTrainer/demo_case/Oscillating_Loss_Case`
$ python demo.py
```


## Results

### Effectiveness
![avatar](https://github.com/shiningrain/tmpfigure/blob/master/ICSE21/Figure3.png)

To evaluate the effectiveness of AUTOTRAINER, we run **495** collected model training scripts to test the effectiveness of AUTOTRAINER. From these models, AUTOTRAINER has detected **262** buggy models and **316** training problems. 
Then AUTORTRAINER tries the candidate solutions and repairs **255** models buggy models, the repaired rate reaches **97.33%**. 

Additionally, the model accuracy improvement distribution of the 255 repaired buggy models is shown in the above figure. The average accuracy improvement reaches **47.08%**. Specifically, over **40.78%** of the models get an increase of
50% and over 50% has an increase between 10% and 30%. The maximum improvement reaches **89.84%**


### Efficiency
![avatar](https://github.com/shiningrain/tmpfigure/blob/master/ICSE21/Figure4.png)

To evaluate the efficiency of AUTOTRAINER, we run all 495 model trainings with and without AUTOTRAINER enabled. 
For normal training, the runtime overhead is closely related to the problem checker frequency which is about to **1%**. The above figure shows how this frequency affect the runtime overhead. It is worth to mention that the runtime overhead on smaller datasets is usually larger (e.g., Blob vs. MNIST in the figure).

For buggy trainings, AUTOTRAINER takes **1.19** more training time on average. We performed a deeper analysis to understand the overhead of individual components, and found that retraining takes over **99%** and the rest two parts (i.e., problem checker and repair) takes less than **1%**. To repair a problem, AUTOTRAINER may try several times, which leads AUTOTRAINER training several models.


## Reproducing results:

1. Download data from [Google Drive](https://drive.google.com/file/d/1AnzEwQZtKXAXA6jo4xGdhRLuAjnUFMLd/view?usp=sharing).
2. Find the model and the corresponding configuration you want to reproduce or test. The models has been saved in different directories which are named by the datasets. Each model and its configuration file, experimental results are placed in a separate subdirectory.
3. Use the `reproduce.py` to test reproduce the our experiments. (make sure you have install the environment and read the 'setup' of [AUTOTRAINER](./AutoTrainer/README.md))

```bash
$ cd AutoTrainer
$ python reproduce.py -mp THE_MODEL_PATH -cp THE_CONFIGURATION_PATH
# the result will be saved in the the `tmp` direction and the output message will be shown on the terminal.
```

## Using your own data?

1. Prepare your model and the training configuration file. The training configuration should be a set saved as a `pkl` file which includes batch size, dataset, max training epoch, loss, optimizer and its parameters. You can refer to [this](./AutoTrainer/demo_case/Gradient_Vanish_Case/config.pkl) to complete the configuration file.
2. Rewrite the `get_dataset()` in `reproduce.py` if you need to use your own dataset. You should add the way to load and preprocess your data.
3. Adust the configuration parameters. These parameters are saved in the params in `reproduce.py` and they are all set the default value which is mentioned in our paper. You can adjust them according to the learning tasks.
4. Run the `reproduce.py` following the [guide](#reproducing-results).
