import torch
import torch.nn as nn
import numpy as np


class TransformerModel(nn.Module):
    def __init__(self, cfg):
        super(TransformerModel, self).__init__()
        self.cfg = cfg

        cate_col_size = len(cfg.cate_cols)
        cont_col_size = len(cfg.cont_cols)

        # category
        self.cate_emb = nn.Embedding(cfg.total_cate_size, cfg.emb_size, padding_idx=0)
        self.cate_proj = nn.Sequential(
            nn.Linear(cfg.emb_size * cfg.cate_col_size * cfg.index_per_step, cfg.hidden_size),
            nn.LayerNorm(cfg.hidden_size),
        )

        # continuous
        self.cont_bn = nn.BatchNorm1d(cfg.cont_col_size)
        self.cont_emb = nn.Sequential(
            nn.Linear(cfg.cont_col_size*cfg.index_per_step, cfg.hidden_size),
            nn.LayerNorm(cfg.hidden_size),
        )

        # combination
        self.comb_proj = nn.Sequential(
            nn.ReLU(),
            nn.Linear(cfg.hidden_size*2, cfg.hidden_size),
            nn.LayerNorm(cfg.hidden_size),
        )
        
        self.config = BertConfig( 
            3, # not used
            hidden_size=cfg.hidden_size,
            num_hidden_layers=cfg.nlayers,
            num_attention_heads=cfg.nheads,
            intermediate_size=cfg.hidden_size,
            hidden_dropout_prob=cfg.dropout,
            attention_probs_dropout_prob=cfg.dropout,
        )
        self.encoder = BertEncoder(self.config)        
        
        def get_reg():
            return nn.Sequential(
            nn.Linear(cfg.hidden_size, cfg.hidden_size),
            nn.LayerNorm(cfg.hidden_size),
            nn.Dropout(cfg.dropout),
            nn.ReLU(),            
            nn.Linear(cfg.hidden_size, cfg.target_size),
        )     
        self.reg_layer = get_reg()
        
    def forward(self, cate_x, cont_x, mask):        
        batch_size = cate_x.size(0)
        half_seq_len = cate_x.size(1) // self.cfg.index_per_step
        
        # category
        cate_emb = self.cate_emb(cate_x).view(batch_size, half_seq_len, -1)
        cate_emb = self.cate_proj(cate_emb)

        # continuous
        cont_x = self.cont_bn(cont_x.view(-1, cont_x.size(-1))).view(batch_size, -1, cont_x.size(-1))
        cont_emb = self.cont_emb(cont_x.view(batch_size, half_seq_len, -1))        
        
        # combination
        seq_emb = torch.cat([cate_emb, cont_emb], 2)        
        seq_emb = self.comb_proj(seq_emb)   
        
        mask, _ = mask.view(batch_size, half_seq_len, -1).max(2)
        mask = mask.unsqueeze(1).unsqueeze(2)
        
        encoded_layers = self.encoder(seq_emb, attention_mask=mask)
        sequence_output = encoded_layers[0]
        sequence_output = sequence_output[:, -1]        
        
        pred_y = self.reg_layer(sequence_output)

        return pred_y


class MultiHeadAttention(nn.Module):
    def __init__(self, cfg, USE_BIAS=True):
        super(MultiHeadAttention,self).__init__()
        if (cfg.d_feat % cfg.n_head) != 0:
            raise ValueError("d_feat(%d) should be divisible by b_head(%d)"%(cfg.d_feat, cfg.nhead))
        self.d_feat = cfg.d_feat
        self.n_head = cfg.n_head
        self.d_head = self.d_feat // self.n_head
        self.sq_d_k = np.sqrt(self.d_head)
        self.dropout = nn.Dropout(p=self.cfg.dropout)

        self.lin_Q = nn.Linear(self.d_feat, self.d_feat, USE_BIAS)  # the input dim needs to be changed to the size of seq_emb
        self.lin_K = nn.Linear(self.d_feat, self.d_feat, USE_BIAS)
        self.lin_V = nn.Linear(self.d_feat, self.d_feat, USE_BIAS)
        self.lin_O = nn.Linear(self.d_feat, self.d_feat, USE_BIAS)
        

    def forward(self, input, mask):
        n_batch = input.shape[0]
        
        Q = self.lin_Q(input)
        K = self.lin_K(input)
        V = self.lin_V(input)

        Q = Q.view(n_batch, -1, self.n_head, self.d_head)
        K = K.view(n_batch, -1, self.n_head, self.d_head)
        V = V.view(n_batch, -1, self.n_head, self.d_head)

        Q = Q.transpose(1, 2)
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)
 
        scores = torch.matmul(Q, K.transpose(-1, -2)) / self.sq_d_k 
        scores = scores.masked_fill(mask == 0, -float('inf'))
        attention = torch.softmax(scores, dim=-1)
        output = torch.matmul(self.dropout(attention), V) 

        output = output.transpose(1, 2).contiguous()
        output = output.view(n_batch, -1, self.d_feat)
        output = self.lin_O(output)

        return output
