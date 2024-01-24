import os
import yaml
import wandb
import datetime
from sklearn.metrics import roc_auc_score
from SaintPlusAlpha.utils import load_config, get_logger, seed_everything, logging_conf
from SaintPlusAlpha.train import train

logger = get_logger(logging_conf)

def main(cfg) :
    wandb.login()
    wandb.init(project = sweep_cfg['project'], config = cfg)
    
    logger.info('Start Training ...')
    train(cfg)

if __name__ == '__main__' :
    cfg = load_config('./config.yaml')
    seed_everything(seed = 42)
    os.makedirs(cfg['model_dir'], exist_ok = True)
    
    yaml_file = cfg['sweep_yaml']
    with open(yaml_file) as f :
        sweep_cfg = yaml.load(f, Loader = yaml.FullLoader)
    sweep_id = wandb.sweep(sweep = sweep_cfg, project = sweep_cfg['project'])
    
    wandb.agent(sweep_id, function = lambda : main(cfg))