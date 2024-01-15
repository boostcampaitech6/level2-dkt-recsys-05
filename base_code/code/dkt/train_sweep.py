import os

import numpy as np
import torch
import wandb

from dkt import trainer_sweep, trainer
from dkt.configs import load_config
from dkt.dataloader import Preprocess
from dkt.utils import get_logger, set_seeds, logging_conf
import yaml

logger = get_logger(logging_conf)

def main(cfg, train_data, valid_data):
    wandb.init(project=sweep_configuration["project"], config=cfg)
    
    logger.info("Building Model ...")
    model: torch.nn.Module = trainer.get_model(cfg=cfg).to(cfg['device'])
    
    logger.info("Start Training ...")
    trainer_sweep.run(cfg=cfg, train_data=train_data, valid_data=valid_data, model=model)


if __name__ == "__main__":
    cfg = load_config("config_sweep.yaml")
    set_seeds(cfg["seed"])

    os.makedirs(cfg["model_dir"], exist_ok=True)
    
    wandb.login()
    yaml_file = cfg["yaml_dir"] + cfg["yaml"]
    
    with open(yaml_file) as f:
        sweep_configuration = yaml.load(f, Loader=yaml.FullLoader)
    sweep_id = wandb.sweep(sweep=sweep_configuration, project=sweep_configuration["project"])
   
    logger.info("Preparing data ...")
    preprocess = Preprocess(cfg)
    preprocess.load_train_data(file_name=cfg["file_name"])
    
    train_data: np.ndarray = preprocess.get_train_data()
    train_data, valid_data = preprocess.split_data(data=train_data)
    
    wandb.agent(sweep_id, function=lambda: main(cfg, train_data, valid_data))