import os
import torch
import random
import numpy as np

def seed_everything(seed : int = 42) :
    random.seed(seed)
    np.random.seed(seed)
    # tf.random.set_seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    torch.use_deterministic_algorithms(True)
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    os.environ['TORCH_USE_CUDA_DSA'] = '1'
    os.environ['TORCH_DSA_CUDBG'] = '1'
    
def load_config(config_file) :
    import yaml
    
    with open(config_file) as file :
        config = yaml.safe_load(file)

    return config


def get_logger(logger_conf : dict) :
    import logging
    import logging.config

    logging.config.dictConfig(logger_conf)
    logger = logging.getLogger()
    return logger


logging_conf = {  # only used when 'user_wandb==False'
    'version' : 1,
    'formatters' : {
        'basic' : {'format' : '%(asctime)s - %(name)s - %(levelname)s - %(message)s'}
    },
    'handlers' : {
        'console' : {
            'class' : 'logging.StreamHandler',
            'level' : 'INFO',
            'formatter' : 'basic',
            'stream' : 'ext://sys.stdout',
        },
        'file_handler' : {
            'class' : 'logging.FileHandler',
            'level' : 'DEBUG',
            'formatter' : 'basic',
            'filename' : 'run.log',
        },
    },
    'root' : {'level' : 'INFO', 'handlers' : ['console', 'file_handler']},
}
