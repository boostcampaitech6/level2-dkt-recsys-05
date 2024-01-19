import numpy as np
import pandas as pd
import time
import pickle
from datetime import datetime
from collections import defaultdict
from .utils import load_config

# 개인별 문제 푸는데 걸린 시간
def Lag_Time(df) :
    time_dict = {}
    lag_time = np.zeros(len(df), dtype = np.float32)
    for idx, col in enumerate(df[['userID', 'Timestamp', 'testID']].values) :
        col[1] = time.mktime(datetime.strptime(col[1],'%Y-%m-%d %H:%M:%S').timetuple())
        
        # 처음 문제 푸는 유저
        if col[0] not in time_dict :
            lag_time[idx] = 0
            time_dict[col[0]] = [col[1], col[2], 0] # last_timestamp, last_testID, last_LagTime
        
        else :
            
            # 이 시험지를 풀어봤다면
            if col[2] == time_dict[col[0]][1] :
                lag_time[idx] = time_dict[col[0]][2]
            
            # 이 시험지를 푼 적 없다면
            else :
                lag_time[idx] = col[1] - time_dict[col[0]][0]
                time_dict[col[0]][0] = col[1]
                time_dict[col[0]][1] = col[2]
                time_dict[col[0]][2] = lag_time[idx]

    df['lag_time'] = lag_time/1000/60 # 분으로 출력
    df['lag_time'] = df['lag_time'].clip(0, 1440) # 1440분(하루) 이상이면 1440으로 변환
    return df['lag_time']

# 문제 푸는데 걸린 시간
def ElapsedTime(df) :
    df = df.sort_values(by = ['userID', 'Timestamp']).reset_index(drop = True)
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    df['elapsed_time'] = df.groupby(['userID', 'testID'], as_index = False)['Timestamp'].diff().shift(0).dt.total_seconds()
    df['elapsed_time'] = df['elapsed_time'].fillna(0)
    df['elapsed_time'] = np.where(df['elapsed_time'] > 3600, np.nan, df['elapsed_time']) # 1시간 이상 결측치 처리
    df['elapsed_time'] = df.groupby('problemID')['elapsed_time'].transform(lambda x : x.fillna(x.mean()))
    df['elapsed_time'] = np.round(df['elapsed_time'])
    return df['elapsed_time']

# 문항 별 평균
def Item_Acc(df) :
    Item_Sum = df['assessmentItemID'].map(df.groupby(['assessmentItemID'])['answerCode'].sum().astype('int32').to_dict())
    Item_Count = df['assessmentItemID'].map(df.groupby(['assessmentItemID'])['answerCode'].count().astype('int32').to_dict())
    df['item_acc'] = Item_Sum / Item_Count
    df['item_acc'] = df['item_acc'].fillna(0)
    return df['item_acc']

# 유저 별 평균
def User_Acc(df) :
    User_Sum = df['userID'].map(df.groupby(['userID'])['answerCode'].sum().astype('int32').to_dict())
    User_Count = df['userID'].map(df.groupby(['userID'])['answerCode'].count().astype('int32').to_dict())
    df['user_acc'] = User_Sum / User_Count
    df['user_acc'] = df['user_acc'].fillna(0)
    return df['user_acc']

# 대분류 별 평균
def testCode_Acc(df) :
    testCode_Sum = df['testCode'].map(df.groupby(['testCode'])['answerCode'].sum().astype('int32').to_dict())
    testCode_Count = df['testCode'].map(df.groupby(['testCode'])['answerCode'].count().astype('int32').to_dict())
    df['testCode_acc'] = testCode_Sum / testCode_Count
    df['testCode_acc'] = df['testCode_acc'].fillna(0)
    return df['testCode_acc']

# 문항 번호 별 평균
def problemID_Acc(df) :
    problemID_Sum = df['problemID'].map(df.groupby(['problemID'])['answerCode'].sum().astype('int32').to_dict())
    problemID_Count = df['problemID'].map(df.groupby(['problemID'])['answerCode'].count().astype('int32').to_dict())
    df['problemID_acc'] = problemID_Sum / problemID_Count
    df['problemID_acc'] = df['problemID_acc'].fillna(0)
    return df['problemID_acc']

# 중분류 별 평균
def Tag_Acc(df) :
    Tag_Sum = df['KnowledgeTag'].map(df.groupby(['KnowledgeTag'])['answerCode'].sum().astype('int32').to_dict())
    Tag_Count = df['KnowledgeTag'].map(df.groupby(['KnowledgeTag'])['answerCode'].count().astype('int32').to_dict())
    df['tag_acc'] = Tag_Sum / Tag_Count
    df['tag_acc'] = df['tag_acc'].fillna(0)
    return df['tag_acc']

# 범주형 변수 인덱싱
def indexing(df, col) :
    col2idx = {v : (k+1) for k, v in enumerate(sorted(df[col].unique()))}
    df[col] = df[col].map(col2idx)
    return df

def Feature_Engineering(df) :
    print('Start Feature Engineering')
    df.index = df.index.astype('uint32')
    
    df['testID'] = df['assessmentItemID'].apply(lambda x : x[1:7])
    df['testCode'] = df['assessmentItemID'].apply(lambda x : x[2:3]).astype('int8')
    df['problemID'] = df['assessmentItemID'].apply(lambda x : x[7:]).astype('int8')

    df['lag_time'] = Lag_Time(df)
    df['elapsed_time'] = ElapsedTime(df)
    df['item_acc'] = Item_Acc(df)
    df['user_acc'] = User_Acc(df)
    df['code_acc'] = testCode_Acc(df)
    df['prob_acc'] = problemID_Acc(df)
    df['tag_acc'] = Tag_Acc(df)
    
    df = indexing(df, 'assessmentItemID')
    df = indexing(df, 'testID')
    df = indexing(df, 'KnowledgeTag')
    
    df['answerCode'] += 1 # Wrong : 1 / Correct : 2
    print('Finish Feature Engineering')
    return df

def grouping(cfg, df, features, save_name) :
    df_group = df[features].groupby('userID').apply(lambda df : (
            df['assessmentItemID'].values,
            df['testID'].values,
            df['testCode'].values,
            df['problemID'].values,
            df['KnowledgeTag'].values,
            df['lag_time'].values,
            df['elapsed_time'].values, 
            df['item_acc'].values,
            df['user_acc'].values,
            df['code_acc'].values,
            df['prob_acc'].values,
            df['tag_acc'].values,
            df['answerCode'].values
        ))
    
    with open(cfg['data_dir'] + f'{save_name}.pkl.zip', 'wb') as p :
        pickle.dump(df_group, p)

def Preprocess(cfg, df, is_train = True) :
    print('Start Preprocess')
    df = Feature_Engineering(df)
    features = ['userID', 'assessmentItemID', 'testID', 'testCode', 'problemID', 'KnowledgeTag',
                'lag_time', 'elapsed_time', 'item_acc', 'user_acc',
                'code_acc', 'prob_acc', 'tag_acc', 'answerCode']
    
    # Train / Valid Data
    if is_train :
        # valid_indices = set(df[df['answerCode'] != -1].index).intersection(set(df.reset_index().groupby('userID', as_index = False).last().set_index('index').index))
        # train_df = df.loc[~df.index.isin(valid_indices)]
        # valid_df = df.loc[df.index.isin(valid_indices)]
        
        valid_size = 0.99
        train_df = df[:int(df.shape[0] * valid_size)]
        valid_df = df[int(df.shape[0] * valid_size):]
        print(f'Train : {train_df.shape}, Valid : {valid_df.shape}')
        print('=' * 50)

        print('Start Train and Valid Data Grouping')
        grouping(cfg, train_df, features, 'Train_SPA')
        grouping(cfg, valid_df, features, 'Valid_SPA')
        print('Finish Preprocess')
    
    # Test Data
    else :
        print('Start Test Data Grouping')
        grouping(cfg, df, features, 'Test_SPA')
        print('Finish Preprocess')


if __name__ == '__main__' :
    cfg = load_config('./config.yaml')
    
    total_data = cfg['data_dir'] + cfg['total_data_name']
    total_df = pd.read_csv(total_data)
    Preprocess(cfg, total_df, True)
    
    test_data = cfg['data_dir'] + cfg['test_data_name']
    test_df = pd.read_csv(test_data)
    Preprocess(cfg, test_df, False)
