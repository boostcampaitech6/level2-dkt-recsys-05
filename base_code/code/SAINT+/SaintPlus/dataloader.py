import numpy as np
import torch.nn as nn
from torch.utils.data import Dataset

class Train_Sequence(Dataset) :
    def __init__(self, groups, seq_len) :
        self.samples = {}
        self.seq_len = seq_len
        self.user_ids = []

        for user_id in groups.index :
            item_id, test_id, lag_time, elapsed_time, item_acc, user_acc, answer_code = groups[user_id]
                
            if len(item_id) < 2 :
                continue

            if len(item_id) > self.seq_len :
                initial = len(item_id) % self.seq_len
                if initial > 2 :
                    self.user_ids.append(f'{user_id}_0')
                    self.samples[f'{user_id}_0'] = (
                        item_id[:initial], test_id[:initial],
                        lag_time[:initial], elapsed_time[:initial],
                        item_acc[:initial], user_acc[:initial],
                        answer_code[:initial]
                    ) 
                chunks = len(item_id) // self.seq_len
                for c in range(chunks) :
                    start = initial + c * self.seq_len
                    end = initial + (c+1) * self.seq_len
                    self.user_ids.append(f'{user_id}_{c+1}')
                    self.samples[f'{user_id}_{c+1}'] = (
                        item_id[start:end], test_id[start:end],
                        lag_time[start:end], elapsed_time[start:end],
                        item_acc[start:end], user_acc[start:end],
                        answer_code[start:end] 
                    ) 
            else :
                self.user_ids.append(f'{user_id}')
                self.samples[f'{user_id}'] = (item_id, test_id, lag_time, elapsed_time, item_acc, user_acc, answer_code)

    def __len__(self) :
        return len(self.user_ids)

    def __getitem__(self, index) :
        user_id = self.user_ids[index]
        item_id, test_id, lag_time, elapsed_time, item_acc, user_acc, answer_code = self.samples[user_id]
        seq_len = len(item_id)

        _item_id = np.zeros(self.seq_len, dtype = int)
        _test_id = np.zeros(self.seq_len, dtype = int)
        _lag_time = np.zeros(self.seq_len, dtype = int)
        _elapsed_time = np.zeros(self.seq_len, dtype = int)
        _item_acc = np.zeros(self.seq_len, dtype = float)
        _user_acc = np.zeros(self.seq_len, dtype = float)
        _answer_code = np.zeros(self.seq_len, dtype = int)
        label = np.zeros(self.seq_len, dtype = int)

        if seq_len == self.seq_len :
            _item_id[:] = item_id
            _test_id[:] = test_id
            _lag_time[:] = lag_time
            _elapsed_time[:] = elapsed_time
            _item_acc[:] = item_acc
            _user_acc[:] = user_acc
            _answer_code[:] = answer_code

        else :
            _item_id[-seq_len:] = item_id
            _test_id[-seq_len:] = test_id
            _lag_time[-seq_len:] = lag_time
            _elapsed_time[-seq_len:] = elapsed_time
            _item_acc[-seq_len:] = item_acc
            _user_acc[-seq_len:] = user_acc
            _answer_code[-seq_len:] = answer_code
        
        _item_id = _item_id[1:]
        _test_id = _test_id[1:]
        _lag_time = _lag_time[1:]
        _elapsed_time = _elapsed_time[1:]
        _item_acc = _item_acc[1:]
        _user_acc = _user_acc[1:]
        label = _answer_code[1:] - 1
        label = np.clip(label, 0, 1)
        _answer_code = _answer_code[:-1]

        return _item_id, _lag_time, _elapsed_time, _item_acc, _user_acc, _answer_code, label # _test_id

class Test_Sequence(Dataset) :
    def __init__(self, groups, seq_len) :
        self.samples = {}
        self.seq_len = seq_len
        self.user_ids = []
        
        for user_id in groups.index :
            item_id, test_id, lag_time, elapsed_time, item_acc, user_acc, answer_code = groups[user_id]
                
            if len(item_id) < 2 :
                continue
            
            if len(item_id) > self.seq_len :
                self.user_ids.append(f'{user_id}')
                self.samples[f'{user_id}'] = (
                    item_id[-seq_len:], test_id[-seq_len:],
                    lag_time[-seq_len:], elapsed_time[-seq_len:],
                    item_acc[-seq_len:], user_acc[-seq_len:],
                    answer_code[-seq_len:])
            else :
                self.user_ids.append(f'{user_id}')
                self.samples[f'{user_id}'] = (item_id, test_id, lag_time, elapsed_time, item_acc, user_acc, answer_code)
    
    def __len__(self) :
        return len(self.user_ids)
    
    def __getitem__(self, index) :
        user_id = self.user_ids[index]
        item_id, test_id, lag_time, elapsed_time, item_acc, user_acc, answer_code = self.samples[user_id]
        seq_len = len(item_id)

        _item_id = np.zeros(self.seq_len, dtype = int)
        _test_id = np.zeros(self.seq_len, dtype = int)
        _lag_time = np.zeros(self.seq_len, dtype = int)
        _elapsed_time = np.zeros(self.seq_len, dtype = int)
        _item_acc = np.zeros(self.seq_len, dtype = float)
        _user_acc = np.zeros(self.seq_len, dtype = float)
        _answer_code = np.zeros(self.seq_len, dtype = int)
        label = np.zeros(self.seq_len, dtype = int)

        if seq_len == self.seq_len :
            _item_id[:] = item_id
            _test_id[:] = test_id
            _lag_time[:] = lag_time
            _elapsed_time[:] = elapsed_time
            _item_acc[:] = item_acc
            _user_acc[:] = user_acc
            _answer_code[:] = answer_code

        else :
            _item_id[-seq_len:] = item_id
            _test_id[-seq_len:] = test_id
            _lag_time[-seq_len:] = lag_time
            _elapsed_time[-seq_len:] = elapsed_time
            _item_acc[-seq_len:] = item_acc
            _user_acc[-seq_len:] = user_acc
            _answer_code[-seq_len:] = answer_code
        
        _item_id = _item_id[1:]
        _test_id = _test_id[1:]
        _lag_time = _lag_time[1:]
        _elapsed_time = _elapsed_time[1:]
        _item_acc = _item_acc[1:]
        _user_acc = _user_acc[1:]
        label = _answer_code[1:] - 1
        label = np.clip(label, 0, 1)
        _answer_code = _answer_code[:-1]

        return _item_id, _lag_time, _elapsed_time, _item_acc, _user_acc, _answer_code, label # _test_id

