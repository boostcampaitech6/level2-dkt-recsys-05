import numpy as np
import time
import pickle
import datetime
import wandb
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score
from .utils import load_config, seed_everything
from .dataloader import Train_Sequence
from .model import SaintPlus, NoamOpt


def save_model(cfg, model) :
    check_point = {
        'ckpt' : model.state_dict()
    }
    torch.save(check_point, cfg['model_dir'] + cfg['model_name'])

def train(cfg) :
    
    wandb.run.name = cfg['runname'] + '_' + str(datetime.datetime.now().strftime('%y%m%d_%H%M%S'))
    wandb.run.save()
    
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
    n_epochs = cfg['n_epochs']
    patience = cfg['patience']
    batch_size = cfg['batch_size']
    # scheduler = cfg['scheduler']
    
    # train, valid 데이터 가져오기
    with open(cfg['data_dir'] + 'Train_SP.pkl.zip', 'rb') as p :
        train_group = pickle.load(p)
    with open(cfg['data_dir'] + 'Valid_SP.pkl.zip', 'rb') as p :
        valid_group = pickle.load(p)

    train_seq = Train_Sequence(train_group, seq_len)
    train_loader = DataLoader(train_seq, batch_size = batch_size, shuffle = True, num_workers = cfg['num_workers'])

    valid_seq = Train_Sequence(valid_group, seq_len)
    valid_loader = DataLoader(valid_seq, batch_size = batch_size, shuffle = False, num_workers = cfg['num_workers'])

    model = SaintPlus(n_layers = n_layers, n_heads = n_heads, d_model = d_model, d_ffn = d_ffn,
                      max_len = max_len, seq_len = seq_len, n_question = n_question, n_test = n_test,
                      dropout = dropout)

    criterion = nn.BCELoss()
    
    if cfg['optimizer'] == 'Adam' :
        opt = NoamOpt(d_model, 1, cfg['warmup_step'], optim.Adam(model.parameters(), lr = cfg['lr']))
    elif cfg['optimizer'] == 'NAdam' :
        opt = NoamOpt(d_model, 1, cfg['warmup_step'], optim.NAdam(model.parameters(), lr = cfg['lr']))
    elif cfg['optimizer'] == 'AdamW' :
        opt = NoamOpt(d_model, 1, cfg['warmup_step'], optim.AdamW(model.parameters(), lr = cfg['lr']))
    else : 
        raise ValueError(f"Unsupported optimizer : {cfg['optimizer']}")
    
    model.to(device)
    criterion.to(device)

    train_losses = []
    valid_losses = []
    valid_aucs = []
    best_auc = 0
    count = 0
    for epoch in range(n_epochs) :
        print(f'============ Epoch {epoch + 1} Training   ============')
        model.train()
        t_s = time.time()
        train_loss = []
        train_labels = []
        train_preds = []
        
        for step, data in enumerate(train_loader) :
            item_id         = data[0].to(device).long()
        #   test_id         = data[1].to(device).long()
            lag_time        = data[1].to(device).float()
            elapsed_time    = data[2].to(device).float()
            item_acc        = data[3].to(device).float()
            user_acc        = data[4].to(device).float()
            answer_correct  = data[5].to(device).long()
            label           = data[6].to(device).float()

            opt.optimizer.zero_grad()
            
            preds = model(item_id, lag_time, elapsed_time, item_acc, user_acc, answer_correct) # test_id
            loss_mask = (answer_correct != 0)
            preds_masked = torch.masked_select(preds, loss_mask)
            label_masked = torch.masked_select(label, loss_mask)
            loss = criterion(preds_masked, label_masked)

            loss.backward()
            opt.step()

            train_loss.append(loss.item())
            train_labels.extend(label_masked.view(-1).data.cpu().numpy())
            train_preds.extend(preds_masked.view(-1).data.cpu().numpy())

        train_loss = np.mean(train_loss)
        train_auc = roc_auc_score(train_labels, train_preds)
        
        print(f'============ Epoch {epoch + 1} Validation ============')
        model.eval()
        valid_loss = []
        valid_labels = []
        valid_preds = []

        for step, data in enumerate(valid_loader) :
            item_id         = data[0].to(device).long()
        #   test_id         = data[1].to(device).long()
            lag_time        = data[1].to(device).float()
            elapsed_time    = data[2].to(device).float()
            item_acc        = data[3].to(device).float()
            user_acc        = data[4].to(device).float()
            answer_correct  = data[5].to(device).long()
            label           = data[6].to(device).float()
            
            preds = model(item_id, lag_time, elapsed_time, item_acc, user_acc, answer_correct) # test_id
            loss_mask = (answer_correct != 0)
            preds_masked = torch.masked_select(preds, loss_mask)
            label_masked = torch.masked_select(label, loss_mask)

            valid_loss.append(loss.item())
            valid_labels.extend(label_masked.view(-1).data.cpu().numpy())
            valid_preds.extend(preds_masked.view(-1).data.cpu().numpy())

        valid_loss = np.mean(valid_loss)
        valid_auc = roc_auc_score(valid_labels, valid_preds)
        
        # EearlyStop Count & Checkpoint
        if valid_auc > best_auc :
            print(f'Epoch {epoch + 1} Save Model.')
            save_model(cfg, model)
            best_auc = valid_auc
            count = 0
        else :
            count += 1
        
        train_losses.append(train_loss)
        valid_losses.append(valid_loss)
        valid_aucs.append(valid_auc)
        
        t_e = int((time.time() - t_s))
        print(f'Train Loss : {train_loss:.5f} / Train AUC : {train_auc:.5f}')
        print(f'Valid Loss : {valid_loss:.5f} / Valid AUC : {valid_auc:.5f}')
        print(f'EarlyStop Count : {count}/{cfg["patience"]} & Train Time {t_e} sec')
        
        wandb.log({'Train Loss' : train_loss,
                   'Train AUC'  : train_auc,
                   'Valid Loss' : valid_loss,
                   'Valid AUC'  : valid_auc,
                   'Best AUC'   : best_auc,
                   'EarlyStop Count' : count})
        
        # EearlyStop
        if count == patience :
            print(f'EarlyStopping. Best AUC : {best_auc}')
            wandb.finish()
            break
    wandb.finish()
    return best_auc

if __name__== '__main__' :
    cfg = load_config('../config.yaml')
    seed_everything(42)
    train(cfg)