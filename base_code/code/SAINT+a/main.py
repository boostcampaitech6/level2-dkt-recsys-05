import numpy as np
import pandas as pd
import os
import wandb
import torch
import datetime
from SaintPlusAlpha.utils import load_config, get_logger, seed_everything, logging_conf
from SaintPlusAlpha.preprocess import Preprocess
from SaintPlusAlpha.train import train
from SaintPlusAlpha.inference import inference

logger = get_logger(logging_conf)

def main(cfg) :
    
    if torch.cuda.is_available() :
        torch.cuda.init()
    
    wandb.login(key = cfg['key'])
    wandb.init(project = cfg['project'], config = cfg)
    
    if not (os.path.exists(cfg['data_dir'] + 'Train_Group.pkl.zip') and \
            os.path.exists(cfg['data_dir'] + 'Valid_Group.pkl.zip') and \
            os.path.exists(cfg['data_dir'] + 'Test_Group.pkl.zip')) :
        
        logger.info('Preparing Data ...')
        
        total_data = cfg['data_dir'] + cfg['total_data_name']
        total_df = pd.read_csv(total_data)
        Preprocess(cfg, total_df, True)
        
        test_data = cfg['data_dir'] + cfg['test_data_name']
        test_df = pd.read_csv(test_data)
        Preprocess(cfg, test_df, False)
    
    else :
        logger.info('Successed Load Data')


    logger.info('Start Training ...')
    train(cfg)
    
    logger.info('Start Inference ...')
    submission = inference(cfg)
    submit = pd.DataFrame({'id' : np.arange(len(submission)),
                           'prediction' : submission})
    submit.to_csv(cfg['submit_dir'] + cfg['submit_name'], index = False)

if __name__ == '__main__' :
    cfg = load_config('./config.yaml')
    seed_everything(seed = 42)
    os.makedirs(cfg['model_dir'], exist_ok = True)
    
    main(cfg)
