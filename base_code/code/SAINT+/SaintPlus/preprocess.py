import numpy as np
import pandas as pd
import time
import pickle
from datetime import datetime
from collections import defaultdict
from .utils import load_config

# 개인별 문제 푸는데 걸린 시간
def get_time_lag(df) :
    time_dict = {}
    time_lag = np.zeros(len(df), dtype = np.float32)
    for idx, col in enumerate(df[['userID', 'Timestamp', 'testId']].values) :
        col[1] = time.mktime(datetime.strptime(col[1],'%Y-%m-%d %H:%M:%S').timetuple())
        # 처음 문제 푸는 유저
        if col[0] not in time_dict :
            time_lag[idx] = 0
            time_dict[col[0]] = [col[1], col[2], 0] # last_timestamp, last_task_container_id, last_lagtime
        
        else :
            
            # 이 시험지를 풀어봤다면
            if col[2] == time_dict[col[0]][1] :
                time_lag[idx] = time_dict[col[0]][2]
            
            # 이 시험지를 푼 적 없다면
            else :
                time_lag[idx] = col[1] - time_dict[col[0]][0]
                time_dict[col[0]][0] = col[1]
                time_dict[col[0]][1] = col[2]
                time_dict[col[0]][2] = time_lag[idx]

    df['time_lag'] = time_lag/1000/60 # convert to miniute
    # 문제푼지 하루가 지났다면 1440(60*24)로 만들어주고, 아니라면 그대로, 0보다 작다면 0으로 만들어줌
    df['time_lag'] = df['time_lag'].clip(0, 1440) 
    return df

# 문제 푸는데 걸린 시간
def duration(df) :
    df = df.sort_values(by=['userID', 'Timestamp']).reset_index(drop=True)
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    df['months'] = df['Timestamp'].dt.month
    df['days'] = df['Timestamp'].dt.day
    df['ts'] = df['Timestamp'].map(pd.Timestamp.timestamp)
    df['prev_ts'] = df.groupby(['userID', 'testId', 'months','days'])['ts'].shift(1)
    df['prev_ts'] = df['prev_ts'].fillna(0)
    df['elapsed'] = np.where(df['prev_ts'] == 0, 0, df['ts'] - df['prev_ts'])

    indexes = df[df['elapsed'] > 1200].index
    df.loc[indexes, 'elapsed'] = 1200
    df = df.drop(['months','days','ts','prev_ts'],axis='columns')
    return df

# 문항별 평균
def make_assess_ratio(df):
    ratio_dict = defaultdict(float)
    grouped_dict = dict(df.groupby('assessmentItemID')['answerCode'].value_counts())
    assess_keys = list(set([x[0] for x in grouped_dict.keys()]))
    for key in assess_keys:
        if grouped_dict.get((key, 1)):
            right = grouped_dict[(key, 1)]
        else:
            right=0
        if grouped_dict.get((key, 0)):
            wrong = grouped_dict[(key, 0)]
        else:
            wrong = 0
        ratio = right / (right + wrong + 1e-10)
        ratio_dict[key] = ratio

    df['assessmentItemAverage'] = df['assessmentItemID'].map(ratio_dict)
    return df

# 유저별 평균
def make_user_ratio(df):
    ratio_dict = defaultdict(float)
    grouped_dict = dict(df.groupby('userID')['answerCode'].value_counts())
    user_keys = list(set([x[0] for x in grouped_dict.keys()]))
    for key in user_keys:
        if grouped_dict.get((key, 1)):
            right = grouped_dict[(key, 1)]
        else:
            right = 0
        if grouped_dict.get((key, 0)):
            wrong = grouped_dict[(key, 0)]
        else:
            wrong = 0
        ratio = right / (right + wrong + 1e-10)
        ratio_dict[key] = ratio

    df['UserAverage'] = df['userID'].map(ratio_dict)
    return df

# 범주형 변수 인덱싱
def indexing(df, col) :
    col2idx = {v : k for k, v in enumerate(sorted(df[col].unique()))}
    df[col] = df[col].map(col2idx)
    return df

def Feature_Engineering(df) :
    print('Start Feature Engineering')
    df.index = df.index.astype('uint32')
    
    # get time_lag feature
    df = get_time_lag(df)
    
    # 문제 푼 시간
    df = duration(df)
    # 문제별 평균
    df = make_assess_ratio(df)
    # 유저별 평균
    df = make_user_ratio(df)
    # 문제번호 1부터 시작으로 바꾸기
    df = indexing(df, 'assessmentItemID')
    df['assessmentItemID'] += 1
    # 시험지번호 1부터 시작으로 바꾸기
    df = indexing(df, 'testId')
    df['testId'] += 1
    # 문제 맞추면 2, 틀리면 1로 바꾸기
    df['answerCode'] += 1
    print('Finish Feature Engineering')
    return df

def grouping(cfg, df, features, save_name) :
    df_group = df[features].groupby('userID').apply(lambda df : (
            df['assessmentItemID'].values,
            df['testId'].values,
            df['time_lag'].values,
            df['elapsed'].values, 
            df['assessmentItemAverage'].values,
            df['UserAverage'].values,
            df['answerCode'].values
        ))
    with open(cfg['data_dir'] + f'{save_name}.pkl.zip', 'wb') as p :
        pickle.dump(df_group, p)

def Preprocess(cfg, df, is_train = True) :
    print('Start Preprocess')
    df = Feature_Engineering(df)
    features = ['userID', 'assessmentItemID', 'testId', 
                'time_lag', 'elapsed',
                'UserAverage', 'assessmentItemAverage',
                'answerCode']
    
    # Train / Valid Data
    if is_train :
        valid_size = cfg['valid_size']
        train_df = df[:int(df.shape[0] * valid_size)]
        valid_df = df[int(df.shape[0] * valid_size):]
        print(f'Train : {train_df.shape}, Valid : {valid_df.shape}')
        print('=' * 50)

        print('Start Train and Valid Data Grouping')
        grouping(cfg, train_df, features, f'Train_Group_{int(valid_size * 100)}')
        grouping(cfg, valid_df, features, f'Valid_Group_{int((1 - valid_size) * 100)}')
        print('Finish Preprocess')
    
    # Test Data
    else :
        print('Start Test Data Grouping')
        grouping(cfg, df, features, 'Test_Group')
        print('Finish Preprocess')


if __name__ == '__main__' :
    cfg = load_config('./config.yaml')
    
    total_data = cfg['data_dir'] + cfg['total_data_name']
    total_df = pd.read_csv(total_data)
    Preprocess(cfg, total_df, True)
    
    test_data = cfg['data_dir'] + cfg['test_data_name']
    test_df = pd.read_csv(test_data)
    Preprocess(cfg, test_df, False)
