import torch
import torch.nn as nn


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
    def __init__(self, embed_size, heads):
        super(MultiHeadAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads

        assert (
            self.head_dim * heads == embed_size
        ), "Embedding size needs to be divisible by heads"

        self.values = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.keys = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.queries = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.fc_out = nn.Linear(heads * self.head_dim, embed_size)

    def forward(self, values, keys, query, mask):
        N = query.shape[0]
        value_len, key_len, query_len = values.shape[1], keys.shape[1], query.shape[1]

        # Split the embedding into self.heads different pieces
        values = values.reshape(N, value_len, self.heads, self.head_dim)
        keys = keys.reshape(N, key_len, self.heads, self.head_dim)
        queries = query.reshape(N, query_len, self.heads, self.head_dim)

        values = self.values(values)
        keys = self.keys(keys)
        queries = self.queries(queries)

        # Attention mechanism
        # ...

        out = self.fc_out(out)
        return out