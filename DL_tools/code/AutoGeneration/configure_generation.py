import sys
sys.path.append('../../data')
sys.path.append('../../utils')
sys.path.append('../../configure')
from utils import *
import numpy as np
from configure import *
import argparse

def gene_config(config_path):
    config = Config(config_path)
    kwargs = config.get_kwargs()
    return kwargs

def optimizer_config(kwargs,config_path):
    config = Config(config_path)
    opt_kwargs = config.get_opt_kwargs(opt_type=kwargs['optimizer'])
    kwargs['opt_kwargs']=opt_kwargs
    return kwargs,kwargs['optimizer']

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Model Generation')
    parser.add_argument('-config','-cfg',default='/data/zxy/DL_tools/DL_tools/code/AutoGeneration/configs/simplednn.conf', help='configuration path')
    parser.add_argument('-config_opt','-cfgo',default='/data/zxy/DL_tools/DL_tools/code/AutoGeneration/configs/optimizer.conf', help='optimizer configuration path')
    parser.add_argument('--amount', '-am', type=int, default=40, help='The number of generated models')
    parser.add_argument('--save_dir', '-f', type=str, default='/data/zxy/DL_tools/DL_tools/code/AutoGeneration/simplednn/config', help='The path to save model')
    args = parser.parse_args()
    for i in range(args.amount):
        print('--------------------------',i+1)
        kwargs=gene_config(args.config)
        kwargs,add_message=optimizer_config(kwargs,args.config_opt)
        save_path(args.save_dir,add_message=add_message,config=kwargs,method='config')
    print('finish')
