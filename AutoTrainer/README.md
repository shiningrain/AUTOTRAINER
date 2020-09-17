# AutoTrainer

## Setup

To install all dependencies, please run:

```bash
$ pip -r requirements.txt
```

## Usage
The main function of AUTOTRAINER is `model_train()`.This function accepts parameters like model, training configuration set as input.  For details, please refer to the corresponding function [comments](./utils/utils.py).  
There are two ways to have a easy start on AUTOTRAINER:
1. You can just modify [`reproduce.py`](./reproduce.py) to load your own dataset and then use your training configuration file and model.
2. You can also call `model_train()` to use AUTOTRAINER. In this method, you need to make sure your codes call this function from the correct file and pass the necessary data to it.