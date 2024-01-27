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

class SaintPlusAlpha(nn.Module) :
    def __init__(self, n_layers, n_heads, d_model, d_ffn, max_len, seq_len, \
        n_question, n_test, n_code, n_prob, n_tag, dropout) :
        
        super(SaintPlusAlpha, self).__init__()
        self.n_heads = n_heads
        self.d_model = d_model
        self.n_question = n_question

        self.pos_emb = nn.Embedding(seq_len, d_model)
        
        self.item_emb = nn.Embedding(n_question + 1, d_model)
        self.test_emb = nn.Embedding(n_test + 1, d_model)
        self.code_emb = nn.Embedding(n_code + 1, d_model)
        self.prob_emb = nn.Embedding(n_prob + 1, d_model)
        self.tag_emb = nn.Embedding(n_tag + 1, d_model)

        self.lagT_emb = nn.Linear(1, d_model, bias = False)
        self.elapT_emb = nn.Linear(1, d_model, bias = False)
        self.itemAcc_emb = nn.Linear(1, d_model, bias = False)
        self.userAcc_emb = nn.Linear(1, d_model, bias = False)
        self.codeAcc_emb = nn.Linear(1, d_model, bias = False)
        self.probAcc_emb = nn.Linear(1, d_model, bias = False)
        self.tagAcc_emb = nn.Linear(1, d_model, bias = False)

        self.answerCorr_emb = nn.Embedding(3, d_model)

        self.emb_dense1 = nn.Linear(6 * d_model, d_model)
        self.emb_dense2 = nn.Linear(8 * d_model, d_model)

        self.transformer = nn.Transformer(d_model = d_model, nhead = n_heads,
                                          num_encoder_layers = n_layers,
                                          num_decoder_layers = n_layers,
                                          dim_feedforward = d_ffn,
                                          dropout = dropout)
        self.layer_norm = nn.LayerNorm(d_model)
        self.FFN = FFN(d_model, d_ffn, dropout = dropout)
        self.final_layer = nn.Linear(d_model, 1)
        

    def forward(self, item_id, test_id, test_cd, prob_id, tag, lag_time, elapsed_time, \
        item_acc, user_acc, code_acc, prob_acc, tag_acc, answer_correct) :
        
        device = item_id.device
        seq_len = item_id.shape[1]

        itemid_emb = self.item_emb(item_id)
        testid_emb = self.test_emb(test_id)
        testcd_emb = self.code_emb(test_cd)
        probid_emb = self.prob_emb(prob_id)
        tag_emb = self.tag_emb(tag)

        lag_time = torch.log(lag_time + 1)
        lag_time = lag_time.view(-1, 1)                             # [batch * seq_len, 1]
        lag_time = self.lagT_emb(lag_time)                          # [batch * seq_len, d_model]
        lag_time = lag_time.view(-1, seq_len, self.d_model)         # [batch,  seq_len, d_model]
        
        elapsed_time = torch.log(elapsed_time + 1)
        elapsed_time = elapsed_time.view(-1, 1)                     # [batch * seq_len, 1]
        elapsed_time = self.elapT_emb(elapsed_time)                 # [batch * seq_len, d_model]
        elapsed_time = elapsed_time.view(-1, seq_len, self.d_model) # [batch,  seq_len, d_model]
        
        item_acc = torch.log(item_acc + 1)
        item_acc = item_acc.view(-1, 1)                             # [batch * seq_len, 1]
        item_acc = self.itemAcc_emb(item_acc)                       # [batch * seq_len, d_model]
        item_acc = item_acc.view(-1, seq_len, self.d_model)         # [batch,  seq_len, d_model]
        
        user_acc = torch.log(user_acc + 1)
        user_acc = user_acc.view(-1, 1)                             # [batch * seq_len, 1]
        user_acc = self.userAcc_emb(user_acc)                       # [batch * seq_len, d_model]
        user_acc = user_acc.view(-1, seq_len, self.d_model)         # [batch,  seq_len, d_model]
        
        code_acc = torch.log(code_acc + 1)
        code_acc = code_acc.view(-1, 1)                             # [batch * seq_len, 1]
        code_acc = self.codeAcc_emb(code_acc)                       # [batch * seq_len, d_model]
        code_acc = code_acc.view(-1, seq_len, self.d_model)         # [batch,  seq_len, d_model]
        
        prob_acc = torch.log(prob_acc + 1)
        prob_acc = prob_acc.view(-1, 1)                             # [batch * seq_len, 1]
        prob_acc = self.probAcc_emb(prob_acc)                       # [batch * seq_len, d_model]
        prob_acc = prob_acc.view(-1, seq_len, self.d_model)         # [batch,  seq_len, d_model]
        
        tag_acc = torch.log(tag_acc + 1)
        tag_acc = tag_acc.view(-1, 1)                               # [batch * seq_len, 1]
        tag_acc = self.tagAcc_emb(tag_acc)                          # [batch * seq_len, d_model]
        tag_acc = tag_acc.view(-1, seq_len, self.d_model)           # [batch,  seq_len, d_model]
        
        answer_correct_emb = self.answerCorr_emb(answer_correct)

        encoder_val = torch.cat((itemid_emb, testid_emb, testcd_emb, probid_emb, tag_emb, lag_time), axis = -1)
        encoder_val = self.emb_dense1(encoder_val)
        decoder_val = torch.cat((lag_time, elapsed_time, item_acc, user_acc, code_acc, prob_acc, tag_acc, answer_correct_emb), axis = -1)
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