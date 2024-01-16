import torch
from torch.utils.data import Dataset

import pandas as pd
import os
import copy
from tqdm import tqdm 


class TransformerDataset(Dataset):
    def __init__(self, df, cfg):        
        
        self.train = cfg.train
        if self.train:
            self.user_id_index_list = cfg.user_id_index_list
            self.start_index_by_user_id = cfg.start_index_by_user_id
            self.len = len(self.user_id_index_list)

        else:
            self.user_id_len = cfg.user_id_len
            self.start_index_by_user_id = cfg.start_index_by_user_id
            self.end_index_by_user_id = cfg.end_index_by_user_id
            self.len = self.user_id_len

        self.max_seq_len = cfg.seq_len
        self.cate_cols = cfg.cate_cols
        self.cont_cols = cfg.cont_cols
        
        self.cate_features = df[self.cate_cols].values
        self.cont_features = df[self.cont_cols].values

        self.device = cfg.device

    def __getitem__(self, idx):
        
        if self.train:
            user_id, end_index = self.user_id_index_list[idx]
            start_index = self.start_index_by_user_id[user_id]
        else:
            end_index = self.end_index_by_user_id[idx]
            start_index = self.start_index_by_user_id[idx]

        end_index += 1
        start_index = max(end_index - self.max_seq_len, start_index)
        seq_len = end_index - start_index

        with torch.device(self.device):
            # 0으로 채워진 output tensor 제작                  
            cate_feature = torch.zeros(self.max_seq_len, len(self.cate_cols), dtype=torch.long)
            cont_feature = torch.zeros(self.max_seq_len, len(self.cont_cols), dtype=torch.float)
            mask = torch.BoolTensor(self.max_seq_len)
        
            # tensor에 값 채워넣기
            cate_feature[-seq_len:] = torch.ShortTensor(self.cate_features[start_index:end_index]) # 16bit signed integer
            cont_feature[-seq_len:] = torch.HalfTensor(self.cont_features[start_index:end_index]) # 16bit float
            mask[:-seq_len] = True
            mask[-seq_len:] = False        
                
            # target은 꼭 cont_feature의 맨 뒤에 놓자
            target = torch.FloatTensor([cont_feature[-1, -1]])

        # data leakage가 발생할 수 있으므로 0으로 모두 채운다
        cont_feature[-1, -1] = 0
        
        return cate_feature.to(self.device), cont_feature.to(self.device), mask.to(self.device), target.to(self.device)
        # return cate_feature, cont_feature, mask, target
        
    def __len__(self):
        return self.len


class PrepareData:
    def __init__(self, cfg):
        self.cfg = cfg

        cfg.cate_col_size = len(cfg.cate_cols)
        cfg.cont_col_size = len(cfg.cont_cols)

        seleted_columns = [column for column in cfg.cate_cols] + [column for column in cfg.cont_cols]

        if cfg.train:
            train_cfg = copy.deepcopy(cfg)
            valid_cfg = copy.deepcopy(train_cfg)
            if not os.path.exists(os.path.join(self.cfg.data_dir, "valid_data.csv")):
                self._split_data()

            train, valid, _, merged = self._load_data()
            self._train_set_variables(train_cfg, valid_cfg, train, valid)
            train, valid = self._indexing_data(train, valid, base=merged)
            self.output = {'train_data' : train[seleted_columns], 
                       'valid_data' : valid[seleted_columns],
                       'train_cfg' : train_cfg,
                       'valid_cfg' : valid_cfg}

        else:
            test_cfg = copy.deepcopy(cfg)
            _, _, test, merged = self._load_data()
            self._test_set_variables(test_cfg, test)
            test = self._indexing_data(test, base=merged)
            test = test[0]
            self.output = {'test_data' : test[seleted_columns],
                           'test_cfg' : test_cfg}


    def _load_data(self) -> pd.DataFrame: 
        path1 = os.path.join(self.cfg.data_dir, "train_data.csv")
        path2 = os.path.join(self.cfg.data_dir, "valid_data.csv")
        path3 = os.path.join(self.cfg.data_dir, "test_data.csv")

        train = pd.read_csv(path1)
        valid = pd.read_csv(path2)
        test = pd.read_csv(path3)

        merged = pd.concat([train, valid, test])

        return train, valid, test, merged
    

    def _indexing_data(self, *datas, **kwargs) -> tuple[pd.DataFrame, dict]:
        cate_cols = self.cfg.cate_cols
        df_mod = [pd.DataFrame() for _ in datas]

        # nan 값이 0이므로 위해 offset은 1에서 출발한다
        cate_offset = 1

        for col in tqdm(cate_cols):

            # 각 column마다 mapper를 만든다
            cate2idx = {}
            for v in kwargs['base'][col].unique():

                # np.nan != np.nan은 True가 나온다
                # nan 및 None은 넘기는 코드
                if (v != v) | (v == None):
                    continue 

                # nan을 고려하여 offset을 추가한다
                cate2idx[v] = len(cate2idx) + cate_offset

            # mapping
            for i, data in enumerate(datas):
                df_mod[i][col] = data[col].map(cate2idx).astype(int)
              

            # 하나의 embedding layer를 사용할 것이므로 다른 feature들이 사용한 index값을
            # 제외하기 위해 offset값을 지속적으로 추가한다
            cate_offset += len(cate2idx)
        
        self.cfg.total_cate_size = cate_offset
        
        for i, data in enumerate(datas):
            df_mod[i][self.cfg.cont_cols] = data[self.cfg.cont_cols]

        return df_mod
    

    def _split_data(self):
        path1 = os.path.join(self.cfg.data_dir, "train_data.csv")
        train = pd.read_csv(path1)
        train.to_csv(os.path.join(self.cfg.data_dir, "train_data.csv.bak"), index=False)

        train = train.reset_index()
        indexes_by_train_users = train.groupby('userID')['index'].agg(list)
        valid = train.copy(deep=False)

        remove_list = []

        for user, indices in indexes_by_train_users.items():
            # 각 사용자별로 인덱스의 20% 계산
            split_idx = int(len(indices) * 0.8)
            
            remove_list.append(indexes_by_train_users[user][split_idx:])

        remove_list = [item for sublist in remove_list for item in sublist]
        train = train[~train['index'].isin(remove_list)]
        train = train.drop(columns=['index']).reset_index(drop=True)

        train.to_csv(os.path.join(self.cfg.data_dir, "train_data.csv"), index=False)
        valid.to_csv(os.path.join(self.cfg.data_dir, "valid_data.csv"), index=False)


    def _train_set_variables(self, train_cfg, valid_cfg, train, valid):
        if not os.path.exists(os.path.join(self.cfg.data_dir, "valid_data.csv")):
            self._split_data()

        indexes_by_train_users = train.reset_index().groupby('userID')['index'].agg(list)
        train_cfg.start_index_by_user_id = indexes_by_train_users.apply(lambda x: x[0])
        train_cfg.user_id_index_list = [(user_id, index)
                                for user_id, indexs in indexes_by_train_users.items()
                                for index in indexs]
        
        indexes_by_valid_users = valid.reset_index().groupby('userID')['index'].agg(list)
        valid_cfg.start_index_by_user_id = indexes_by_valid_users.apply(lambda x: x[0])
        for user, train_indices in indexes_by_train_users.items():
            indexes_by_valid_users[user] = indexes_by_valid_users[user][len(train_indices):]

        valid_cfg.user_id_index_list = [(user_id, index)
                                for user_id, indexs in indexes_by_valid_users.items()
                                for index in indexs]

    def _test_set_variables(self, cfg, test):
        indexes_by_users = test.reset_index().groupby('userID')['index'].agg(list)

        cfg.user_id_len = len(test['userID'].unique())
        cfg.start_index_by_user_id = indexes_by_users.apply(lambda x: x[0]).tolist()
        cfg.end_index_by_user_id = indexes_by_users.apply(lambda x: x[-1]).tolist()


    def get_data(self):
        return self.output
