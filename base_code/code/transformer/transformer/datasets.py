import torch
from torch.utils.data import Dataset

import pandas as pd
import os


class TransformerDataset(Dataset):
    def __init__(self, df, cfg, device, max_seq_len=100, max_content_len=1000):        
        
        self.max_seq_len = max_seq_len
        self.max_content_len = max_content_len
        self.cfg = cfg

        if self.cfg.mode == 'train':
            self.user_id_index_list = cfg.user_id_index_list
            self.start_index_by_user_id = cfg.start_index_by_user_id
            self.len = len(self.user_id_index_list)

        else:
            self.user_id_len = cfg.user_id_len
            self.start_index_by_user_id = cfg.start_index_by_user_id
            self.end_index_by_user_id = cfg.end_index_by_user_id
            self.len = self.user_id_len

        self.cate_cols = cfg.cate_cols
        self.cont_cols = cfg.cont_cols
        
        self.cate_features = df[self.cate_cols].values
        self.cont_features = df[self.cont_cols].values

        self.device = device

    def __getitem__(self, idx):
        
        if self.cfg.mode == 'train':
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
            mask = torch.zeros(self.max_seq_len, dtype=torch.int16)
        
            # tensor에 값 채워넣기
            cate_feature[-seq_len:] = torch.ShortTensor(self.cate_features[start_index:end_index]) # 16bit signed integer
            cont_feature[-seq_len:] = torch.HalfTensor(self.cont_features[start_index:end_index]) # 16bit float
            mask[-seq_len:] = 1        
                
            # target은 꼭 cont_feature의 맨 뒤에 놓자
            target = torch.cuda.FloatTensor([cont_feature[-1, -1]])

        # data leakage가 발생할 수 있으므로 0으로 모두 채운다
        cont_feature[-1, -1] = 0
        
        # return cate_feature.to(self.device), cont_feature.to(self.device), mask.to(self.device), target.to(self.device)
        return cate_feature, cont_feature, mask, target
        
    def __len__(self):
        return self.len


class PrepareData:
    def __init__(self, cfg):
        self.cfg = cfg
        self.merged, self.df = self._load_data()
        self.indexes_by_users = self.df.reset_index().groupby('userID')['index'].apply(lambda x: x.values)

        cfg.cate_col_size = len(cfg.cate_cols)
        cfg.cont_col_size = len(cfg.cont_cols)

        if self.cfg.mode == 'train':
            cfg.user_id_index_list = [(user_id, index)
                                    for user_id, indexs in self.indexes_by_users.items()
                                    for index in indexs]
            cfg.start_index_by_user_id = self.indexes_by_users.apply(lambda x: x[0])

        else:
            cfg.user_id_len = len(self.df['userID'].unique())
            cfg.start_index_by_user_id = self.indexes_by_users.apply(lambda x: x[0]).tolist()
            cfg.end_index_by_user_id = self.indexes_by_users.apply(lambda x: x[-1]).tolist()

        df_mod, cfg.total_cate_size = self._indexing_data()
        self.df[df_mod.columns] = df_mod[df_mod.columns]


    def _load_data(self) -> pd.DataFrame: 
        path1 = os.path.join(self.cfg.data_dir, "train_data.csv")
        path2 = os.path.join(self.cfg.data_dir, "test_data.csv")
        train = pd.read_csv(path1)
        test = pd.read_csv(path2)

        data = pd.concat([train, test])

        if self.cfg.mode == 'train':
            df = train
        else:
            df = test
        return data, df
    

    def _indexing_data(self) -> tuple[pd.DataFrame, dict]:
        cate_cols = self.cfg.cate_cols
        mappers_dict = {}

        df_mod = pd.DataFrame()
        # nan 값이 0이므로 위해 offset은 1에서 출발한다
        cate_offset = 1

        for col in cate_cols:

            # 각 column마다 mapper를 만든다
            cate2idx = {}
            for v in self.merged[col].unique():

                # np.nan != np.nan은 True가 나온다
                # nan 및 None은 넘기는 코드
                if (v != v) | (v == None):
                    continue 

                # nan을 고려하여 offset을 추가한다
                cate2idx[v] = len(cate2idx) + cate_offset

            # mapping
            df_mod[col] = self.df[col].map(cate2idx).astype(int)

            # 하나의 embedding layer를 사용할 것이므로 다른 feature들이 사용한 index값을
            # 제외하기 위해 offset값을 지속적으로 추가한다
            cate_offset += len(cate2idx)
        
        return df_mod, cate_offset


    def get_data(self):
        return self.df
