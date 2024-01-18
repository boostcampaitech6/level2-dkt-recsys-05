import numpy as np
import torch.nn as nn
from torch.utils.data import Dataset

class Train_Sequence(Dataset) :
    def __init__(self, groups, seq_len) :
        self.samples = {}
        self.seq_len = seq_len
        self.user_ids = []

        for user_id in groups.index :
            item_id, test_id, t_lag, elaps_ed, assessmentItemAverage,UserAverage,answer_code=groups[user_id]
            if len(item_id) < 2 :
                continue

            if len(item_id) > self.seq_len:
                initial = len(item_id) % self.seq_len
                if initial > 2:
                    self.user_ids.append(f"{user_id}_0")
                    self.samples[f"{user_id}_0"] = (
                        item_id[:initial],test_id[:initial],t_lag[:initial],
                        elaps_ed[:initial], assessmentItemAverage[:initial],
                        UserAverage[:initial],
                        answer_code[:initial]
                    ) 
                chunks = len(item_id)//self.seq_len
                for c in range(chunks):
                    start = initial + c*self.seq_len
                    end = initial + (c+1)*self.seq_len
                    self.user_ids.append(f"{user_id}_{c+1}")
                    self.samples[f"{user_id}_{c+1}"] = (
                        item_id[start:end],test_id[start:end],t_lag[start:end],
                        elaps_ed[start:end],assessmentItemAverage[start:end],
                        UserAverage[start:end],
                        answer_code[start:end] 
                    ) 
            else:
                self.user_ids.append(f"{user_id}")
                self.samples[f"{user_id}"] = (item_id,test_id,t_lag,elaps_ed,assessmentItemAverage,UserAverage,answer_code)

    def __len__(self):
        return len(self.user_ids)

    def __getitem__(self, index):
        user_id = self.user_ids[index]
        item_id, test_id, t_lag, elaps_ed, item_aver,user_aver, answer_code = self.samples[user_id]
        seq_len = len(item_id)

        itemid = np.zeros(self.seq_len, dtype=int)
        testid = np.zeros(self.seq_len, dtype=int)
        time_lag=np.zeros(self.seq_len, dtype=int)
        elapsed= np.zeros(self.seq_len, dtype=int)
        itemaver=np.zeros(self.seq_len, dtype=float)
        useraver=np.zeros(self.seq_len, dtype=float)
        answercode= np.zeros(self.seq_len, dtype=int)
        label = np.zeros(self.seq_len, dtype=int)

        if seq_len == self.seq_len:
            itemid[:] = item_id
            testid[:] = test_id
            time_lag[:]=t_lag
            elapsed[:] = elaps_ed
            itemaver[:] = item_aver
            useraver[:] = user_aver
            answercode[:] = answer_code

        else:
            itemid[-seq_len:] = item_id
            testid[-seq_len:] = test_id
            time_lag[-seq_len:] = t_lag
            elapsed[-seq_len:] = elaps_ed
            itemaver[-seq_len:] = item_aver
            useraver[-seq_len:] = user_aver
            answercode[-seq_len:] = answer_code

        itemid=itemid[1:]
        testid=testid[1:]
        time_lag=time_lag[1:]
        elapsed=elapsed[1:]
        itemaver=itemaver[1:]
        useraver=useraver[1:]
        label = answercode[1:] - 1
        label = np.clip(label, 0, 1)
        answercode = answercode[:-1]

        return itemid, time_lag, elapsed, itemaver,useraver, answercode, label

class Test_Sequence(Dataset):
    def __init__(self, groups, seq_len):
        self.samples = {}
        self.seq_len = seq_len
        self.user_ids = []
        
        for user_id in groups.index:
            # c_id, part, t_c_id, t_lag, q_et, ans_c, q_he, u_ans = groups[user_id]
            item_id, test_id, t_lag, elaps_ed, assessmentItemAverage,UserAverage,answer_code=groups[user_id] # UserAverage
            if len(item_id) < 2:
                continue
            
            if len(item_id) > self.seq_len:
                self.user_ids.append(f"{user_id}")
                self.samples[f"{user_id}"] = (
                    item_id[-seq_len:],test_id[-seq_len:],t_lag[-seq_len:],
                    elaps_ed[-seq_len:],assessmentItemAverage[-seq_len:],
                    UserAverage[-seq_len:],
                    answer_code[-seq_len:])
            else:
                self.user_ids.append(f"{user_id}")
                self.samples[f"{user_id}"] = (item_id,test_id,t_lag,elaps_ed,assessmentItemAverage,UserAverage,answer_code)
    
    def __len__(self):
        return len(self.user_ids)
    
    def __getitem__(self, index):
        user_id = self.user_ids[index]
        item_id, test_id, t_lag, elaps_ed, item_aver,user_aver, answer_code = self.samples[user_id] # user_aver
        seq_len = len(item_id)

        itemid = np.zeros(self.seq_len, dtype=int)
        testid = np.zeros(self.seq_len, dtype=int)
        time_lag=np.zeros(self.seq_len, dtype=int)
        elapsed= np.zeros(self.seq_len, dtype=int)
        itemaver= np.zeros(self.seq_len, dtype=float)
        useraver=np.zeros(self.seq_len, dtype=float)
        answercode= np.zeros(self.seq_len, dtype=int)
        label = np.zeros(self.seq_len, dtype=int)

        if seq_len == self.seq_len:
            itemid[:] = item_id
            testid[:] = test_id
            time_lag[:]=t_lag
            elapsed[:] = elaps_ed
            itemaver[:]=item_aver
            useraver[:]=user_aver
            answercode[:] = answer_code
        else:
            itemid[-seq_len:] = item_id
            testid[-seq_len:] = test_id
            time_lag[-seq_len:] = t_lag
            elapsed[-seq_len:] = elaps_ed
            itemaver[-seq_len:]=item_aver
            useraver[-seq_len:]=user_aver
            answercode[-seq_len:] = answer_code

        itemid=itemid[1:]
        testid=testid[1:]
        time_lag=time_lag[1:]
        elapsed=elapsed[1:]
        itemaver=itemaver[1:]
        useraver=useraver[1:]
        label = answercode[1:] - 1
        label = np.clip(label, 0, 1)
        answercode = answercode[:-1]
        
        return itemid, time_lag, elapsed, itemaver,useraver,answercode, label # useraver

