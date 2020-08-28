import os
import configparser
import numpy as np
import itertools
import copy
import uuid
import time
import pickle
import keras

class Config:
    def __init__(self, cfg_file):
        self.cfg = configparser.ConfigParser()
        self.cfg.read(cfg_file)
        sec=self.cfg.sections()
        #print(1)

    def get_kwargs(self):
        kwargs = {}
        params = self.cfg['parameters']
        for key, value in params.items():
            opts = value.lstrip().rstrip().split(' ')
            opt = np.random.choice(opts)
            opt_value = self._parse_rule(opt)
            kwargs[key] = opt_value
        return kwargs
    
    def get_opt_kwargs(self,opt_type):
        kwargs = {}
        params = self.cfg[opt_type]
        for key, value in params.items():
            opts = value.lstrip().rstrip().split(' ')
            opt = np.random.choice(opts)
            opt_value = self._parse_rule(opt)
            kwargs[key] = opt_value
        return kwargs

    def _parse_rule(self, opt):
        if opt.startswith('intchoice'):
            value = opt.split('-')[-1]
            return int(value)
        elif opt.startswith('int'):
            tmp = opt.split('-')[-1]
            value=int((10**(int(tmp)+1))*round(np.random.rand(),1))
            return value
        elif opt.startswith('str'):
            value = opt.split('-')[-1]
            return value
        elif opt.startswith('floatchoice'):
            tmp = opt.split('-')[-1]
            power=int(-np.random.randint(0,int(tmp)+1))
            value=10**power
            return value
        elif opt.startswith('float'):
            tmp = opt.split('-')[-1]
            value=(10**(int(tmp)-1))*round(np.random.rand(),4)
            return value

def save_path(file_dir,add_message,model=None,config=None,method='model'):
    file_id = str(uuid.uuid3(uuid.NAMESPACE_DNS,str(time.time())))
    if not os.path.exists(file_dir):
        os.makedirs(file_dir)
    if method=='config':
        file_name = 'config_'+add_message+'_'+file_id+'.pkl'
        file_path=os.path.join(file_dir,file_name)
        with open(file_path, 'wb') as f:
            pickle.dump(config, f)
    if method=='model':
        length=len(model.layers)
        model_name = 'model'+add_message+str(length)+'_'+file_id+'.h5'
        file_path=os.path.join(file_dir,model_name)
        model.save(file_path,model)