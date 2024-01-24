import numpy as np
import pandas as pd
import pickle
import torch
from torch.utils.data import DataLoader
from .utils import load_config, seed_everything
from .dataloader import Test_Sequence
from .model import SaintPlus


def inference(cfg) :
    seed_everything(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    n_layers = cfg['n_layers']
    n_heads = cfg['n_heads']
    d_model = cfg['d_model']
    d_ffn = d_model * 4
    max_len = cfg['max_len']
    seq_len = cfg['seq_len']
    n_question = cfg['n_question']
    n_test = cfg['n_test']

    dropout = cfg['dropout']
    batch_size = cfg['batch_size']

    with open(cfg['data_dir'] + 'Test_SP.pkl.zip', 'rb') as p :
        test_group = pickle.load(p)

    test_seq = Test_Sequence(test_group, seq_len)
    test_loader = DataLoader(test_seq, batch_size = batch_size, shuffle = False, num_workers = cfg['num_workers'])

    model = SaintPlus(n_layers = n_layers, n_heads = n_heads, d_model = d_model, d_ffn = d_ffn,
                      max_len = max_len, seq_len = seq_len, n_question = n_question, n_test = n_test,
                      dropout = dropout)

    checkpoint = torch.load(cfg['model_dir'] + cfg['model_name'], map_location = device)
    state_dict = checkpoint['ckpt']
    model.load_state_dict(state_dict)
    model.to(device)
    
    model.eval()
    submission = []
    for step, data in enumerate(test_loader) :
        item_id         = data[0].to(device).long()
    #   test_id         = data[1].to(device).long()
        lag_time        = data[1].to(device).float()
        elapsed_time    = data[2].to(device).float()
        item_acc        = data[3].to(device).float()
        user_acc        = data[4].to(device).float()
        answer_correct  = data[5].to(device).long()
        
        preds = model(item_id, lag_time, elapsed_time, item_acc, user_acc, answer_correct) # test_id
        
        pred = preds[:, -1]
        submission.extend(pred.data.cpu().numpy())
    print(len(submission))
    print('Finish Inference.')
    return submission

if __name__ == '__main__' :
    cfg = load_config('../config.yaml')
    submission = inference(cfg)
    submit = pd.DataFrame({'id' : np.arange(len(submission)),
                           'prediction' : submission})
    submit.to_csv(cfg['submit_dir'] + cfg['submit_name'], index = False)