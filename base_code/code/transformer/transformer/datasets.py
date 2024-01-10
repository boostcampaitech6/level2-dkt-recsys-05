import torch
from torch.utils.data import Dataset

import pandas as pd
import os


class TransformerDataset(Dataset):
    def __init__(self, df, cfg, device, max_seq_len=100, max_content_len=1000):        
        
        self.max_seq_len = max_seq_len
        self.max_content_len = max_content_len
        
        self.user_id_index_list = cfg.user_id_index_list
        self.start_index_by_user_id = cfg.start_index_by_user_id

        self.cate_cols = cfg.cate_cols
        self.cont_cols = cfg.cont_cols
        
        self.cate_features = df[self.cate_cols].values
        self.cont_features = df[self.cont_cols].values

        self.device = device

    def __getitem__(self, idx):
        
        # end_index 추출
        user_id, end_index = self.user_id_index_list[idx]
        end_index += 1
        
        # start_index 계산
        start_index = self.start_index_by_user_id[user_id]
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
        return len(self.user_id_index_list)


class PrepareForTF:
    def __init__(self, cfg):
        self.cfg = cfg
        self.merged, self.train, self.test = self._load_data()

        self.indexes_by_users = self.train.reset_index().groupby('userID')['index'].apply(lambda x: x.values)
        cfg.user_id_index_list = [(user_id, index)
                                  for user_id, indexs in self.indexes_by_users.items()
                                  for index in indexs]
        cfg.start_index_by_user_id = self.indexes_by_users.apply(lambda x: x[0])

        train_mod, test_mod, cfg.mappers_dict, cfg.total_cate_size = self._indexing_data()
        self.train[train_mod.columns] = train_mod[train_mod.columns]
        self.test[test_mod.columns] = test_mod[test_mod.columns]


    def _load_data(self) -> pd.DataFrame: 
        path1 = os.path.join(self.cfg.data_dir, "train_data.csv")
        path2 = os.path.join(self.cfg.data_dir, "test_data.csv")
        train = pd.read_csv(path1)
        test = pd.read_csv(path2)

        data = pd.concat([train, test])
        return data, train, test
    

    def _indexing_data(self) -> tuple[pd.DataFrame, dict]:
        cate_cols = self.cfg.cate_cols
        mappers_dict = {}

        train_mod = pd.DataFrame()
        test_mod = pd.DataFrame()
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

            mappers_dict[col] = cate2idx

            # mapping
            train_mod[col] = self.train[col].map(cate2idx).astype(int)
            test_mod[col] = self.test[col].map(cate2idx).astype(int)

            # 하나의 embedding layer를 사용할 것이므로 다른 feature들이 사용한 index값을
            # 제외하기 위해 offset값을 지속적으로 추가한다
            cate_offset += len(cate2idx)
        
        return train_mod, test_mod, mappers_dict, cate_offset


    def get_data(self):
        return {'train'                     : self.train, 
                'test'                      : self.test, 
                }


class PrepareForIF:
    def __init__(self, cfg):
        self.cfg = cfg
        self.merged, self.train, self.test = self._load_data()

        self.indexes_by_users = self.test.reset_index().groupby('userID')['index'].apply(lambda x: x.values)
        cfg.user_id_len = len(self.test['userID'].unique())
        cfg.start_index_by_user_id = self.indexes_by_users.apply(lambda x: x[0])
        cfg.end_index_by_user_id = self.indexes_by_users.apply(lambda x: x[-1])

        train_mod, test_mod, cfg.mappers_dict, cfg.total_cate_size = self._indexing_data()
        self.train[train_mod.columns] = train_mod[train_mod.columns]
        self.test[test_mod.columns] = test_mod[test_mod.columns]


    def _load_data(self) -> pd.DataFrame: 
        path1 = os.path.join(self.cfg.data_dir, "train_data.csv")
        path2 = os.path.join(self.cfg.data_dir, "test_data.csv")
        train = pd.read_csv(path1)
        test = pd.read_csv(path2)

        data = pd.concat([train, test])
        return data, train, test
    

    def _indexing_data(self) -> tuple[pd.DataFrame, dict]:
        cate_cols = self.cfg.cate_cols
        mappers_dict = {}

        train_mod = pd.DataFrame()
        test_mod = pd.DataFrame()
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

            mappers_dict[col] = cate2idx

            # mapping
            train_mod[col] = self.train[col].map(cate2idx).astype(int)
            test_mod[col] = self.test[col].map(cate2idx).astype(int)

            # 하나의 embedding layer를 사용할 것이므로 다른 feature들이 사용한 index값을
            # 제외하기 위해 offset값을 지속적으로 추가한다
            cate_offset += len(cate2idx)
        
        return train_mod, test_mod, mappers_dict, cate_offset


    def get_data(self):
        return {'train'                     : self.train, 
                'test'                      : self.test, 
                }


class InferenceDataset():
    def __init__(self, df, cfg, device, max_seq_len=100, max_content_len=1000):        
        
        self.max_seq_len = max_seq_len
        self.max_content_len = max_content_len
        
        self.user_id_len = cfg.user_id_len
        self.start_index_by_user_id = cfg.start_index_by_user_id
        self.end_index_by_user_id = cfg.end_index_by_user_id

        self.cate_cols = cfg.cate_cols
        self.cont_cols = cfg.cont_cols
        
        self.cate_features = df[self.cate_cols].values
        self.cont_features = df[self.cont_cols].values

        self.device = device

    def __getitem__(self, idx):
        
        # end_index 추출
        end_index = self.end_index_by_user_id.iloc[idx]
        end_index += 1
        
        # start_index 계산
        start_index = self.start_index_by_user_id.iloc[idx]
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
        return self.user_id_len