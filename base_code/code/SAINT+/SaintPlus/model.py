import torch
import torch.nn as nn
import numpy as np

'''
Reference:
https://arxiv.org/abs/2002.07033
'''

class FFN(nn.Module) :
    def __init__(self, d_model, d_ffn, dropout) :
        super(FFN, self).__init__()
        self.linear1 = nn.Linear(d_model, d_ffn) # [batch, seq_len, ffn_dim]
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(d_ffn, d_model) # [batch, seq_len, d_model]
        self.dropout = nn.Dropout(dropout)

    def forward(self, x) :
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        return self.dropout(x)

class SaintPlus(nn.Module) :
    def __init__(self, n_layers, n_heads, d_model, d_ffn, max_len, seq_len, n_question, n_test, dropout) :
        super(SaintPlus, self).__init__()
        self.n_heads = n_heads
        self.d_model = d_model
        self.n_question = n_question

        self.pos_emb = nn.Embedding(seq_len, d_model)
        
        self.item_emb = nn.Embedding(n_question + 1, d_model)
        self.test_emb = nn.Embedding(n_test + 1, d_model)

        self.lagT_emb = nn.Linear(1, d_model, bias = False)
        self.elapT_emb = nn.Linear(1, d_model, bias = False)
        self.itemAcc_emb = nn.Linear(1, d_model, bias = False)
        self.userAcc_emb = nn.Linear(1, d_model, bias = False)

        self.answerCorr_emb = nn.Embedding(3, d_model)

        self.emb_dense1 = nn.Linear(2 * d_model, d_model)
        self.emb_dense2 = nn.Linear(5 * d_model, d_model)

        self.transformer = nn.Transformer(d_model = d_model, nhead = n_heads,
                                          num_encoder_layers = n_layers,
                                          num_decoder_layers = n_layers,
                                          dim_feedforward = d_ffn,
                                          dropout = dropout)
        self.layer_norm = nn.LayerNorm(d_model)
        self.FFN = FFN(d_model, d_ffn, dropout = dropout)
        self.final_layer = nn.Linear(d_model, 1)

    def forward(self, item_id, lag_time, elapsed_time, item_avg, user_avg, answer_correct) :
        device = item_id.device
        seq_len = item_id.shape[1]

        item_emb = self.item_emb(item_id)
        # test_emb = self.test_emb(test_id)

        lag_time = torch.log(lag_time + 1)
        lag_time = lag_time.view(-1, 1)                             # [batch * seq_len, 1]
        lag_time = self.lagT_emb(lag_time)                          # [batch * seq_len, d_model]
        lag_time = lag_time.view(-1, seq_len, self.d_model)         # [batch,  seq_len, d_model]
        
        elapsed_time = torch.log(elapsed_time + 1)
        elapsed_time = elapsed_time.view(-1, 1)                     # [batch * seq_len, 1]
        elapsed_time = self.elapT_emb(elapsed_time)                 # [batch * seq_len, d_model]
        elapsed_time = elapsed_time.view(-1, seq_len, self.d_model) # [batch,  seq_len, d_model]
        
        item_avg = torch.log(item_avg + 1)
        item_avg = item_avg.view(-1, 1)                             # [batch * seq_len, 1]
        item_avg = self.itemAcc_emb(item_avg)                       # [batch * seq_len, d_model]
        item_avg = item_avg.view(-1, seq_len, self.d_model)         # [batch,  seq_len, d_model]
        
        user_avg = torch.log(user_avg + 1)
        user_avg = user_avg.view(-1, 1)                             # [batch * seq_len, 1]
        user_avg = self.userAcc_emb(user_avg)                       # [batch * seq_len, d_model]
        user_avg = user_avg.view(-1, seq_len, self.d_model)         # [batch,  seq_len, d_model]
        
        answer_correct_emb = self.answerCorr_emb(answer_correct)

        encoder_val = torch.cat((item_emb, lag_time), axis = -1)
        encoder_val = self.emb_dense1(encoder_val)
        decoder_val = torch.cat((lag_time, elapsed_time, item_avg, user_avg, answer_correct_emb), axis = -1)
        decoder_val = self.emb_dense2(decoder_val)

        pos = torch.arange(seq_len).unsqueeze(0).to(device)
        pos_emb = self.pos_emb(pos)
        encoder_val += pos_emb
        decoder_val += pos_emb

        over_head_mask = torch.from_numpy(np.triu(np.ones((seq_len, seq_len)), k = 1).astype('bool'))
        over_head_mask = over_head_mask.to(device)

        encoder_val = encoder_val.permute(1, 0, 2)
        decoder_val = decoder_val.permute(1, 0, 2)
        decoder_val = self.transformer(encoder_val, decoder_val, src_mask = over_head_mask, tgt_mask = over_head_mask, memory_mask = over_head_mask)

        decoder_val = self.layer_norm(decoder_val)
        decoder_val = decoder_val.permute(1, 0, 2)

        final_out = self.FFN(decoder_val)
        final_out = self.layer_norm(final_out + decoder_val)
        final_out = self.final_layer(final_out)
        final_out = torch.sigmoid(final_out)
        return final_out.squeeze(-1)


# Transformer의 Optimizer Learning Rate 조절
# Reference : https://lee-soohyun.tistory.com/262
# https://www.quantumdl.com/entry/11%EC%A3%BC%EC%B0%A82-Attention-is-All-You-Need-Transformer

class NoamOpt :
    'Optim wrapper that implements rate.'
    def __init__(self, model_size, factor, warmup, optimizer) :
        self.optimizer = optimizer
        self._step = 0
        self.warmup = warmup
        self.factor = factor
        self.model_size = model_size
        self._rate = 0

    def step(self) :
        'Update parameters and rate'
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups :
            p['lr'] = rate
        self._rate = rate
        self.optimizer.step()

    def rate(self, step = None) :
        'Implement `lrate` above'
        if step is None :
            step = self._step
        return self.factor * \
            (self.model_size ** (-0.5) *
            min(step ** (-0.5), step * self.warmup ** (-1.5)))