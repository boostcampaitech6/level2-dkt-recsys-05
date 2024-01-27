import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler
import pickle
from .utils import load_config

# 개인별 테스트를 푸는데 걸린 시간
def LagTime(df) :
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    time_dict = {}
    lag_time = np.zeros(len(df), dtype = np.float32)
    for idx, (user, time, test) in enumerate(df[['userID', 'Timestamp', 'testId']].values) :

        if user not in time_dict :
            lag_time[idx] = 0
            time_dict[user] = [time, test, 0]

        else :
            if test == time_dict[user][1] :
                lag_time[idx] = time_dict[user][2]
            else :
                lag_time[idx] = (time - time_dict[user][0]).total_seconds()
                time_dict[user][0] = time
                time_dict[user][1] = test
                time_dict[user][2] = lag_time[idx]
    df['lag_time'] = lag_time / 1000 / 60 # 분 단위로 변환
    df['lag_time'] = df['lag_time'].clip(0, 1440)
    return df['lag_time']

# 문제 푸는데 걸린 시간
def ElapsedTime(df) :
    df = df.sort_values(by = ['userID', 'Timestamp']).reset_index(drop = True)
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    df['elapsed_time'] = df.groupby(['userID', 'testId'], as_index = False)['Timestamp'].diff().shift(0).dt.total_seconds()
    df['elapsed_time'] = df['elapsed_time'].fillna(0)
    df['elapsed_time'] = np.where(df['elapsed_time'] > 1200, np.nan, df['elapsed_time'])
    
    df['problemID'] = df['assessmentItemID'].apply(lambda x : x[7:]).astype('int8')
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

# 범주형 변수 인덱싱
def indexing(df, col) :
    col2idx = {v : (k+1) for k, v in enumerate(df[col].unique())}
    df[col] = df[col].map(col2idx)
    return df[col]

def Feature_Engineering(df) :
    df = df.sort_values(by = ['userID', 'Timestamp']).reset_index(drop = True)
    df.index = df.index.astype('uint32')

    # Feature 생성
    df['lag_time'] = LagTime(df)
    df['elapsed_time'] = ElapsedTime(df)
    df['item_acc'] = Item_Acc(df)
    df['user_acc'] = User_Acc(df)
    
    # Indexing
    df['assessmentItemID'] = indexing(df, 'assessmentItemID')
    df['testId'] = indexing(df, 'testId')
    df['answerCode'] += 1 # Wrong : 1 / Correct : 2
    return df

def grouping(cfg, df, features, save_name) :
    print('User 수 : ', df['userID'].nunique())
    df_group = df[features].groupby('userID').apply(lambda r : (
            r['assessmentItemID'].values,
            r['testId'].values,
            r['lag_time'].values,
            r['elapsed_time'].values, 
            r['item_acc'].values,
            r['user_acc'].values,
            r['answerCode'].values
        ))
    
    with open(cfg['data_dir'] + f'{save_name}.pkl.zip', 'wb') as p :
        pickle.dump(df_group, p)

def Preprocess(cfg, total, test, scaling = False) :
    print('Start Preprocess')
    total = Feature_Engineering(total)
    test  = Feature_Engineering(test)
    features = ['userID', 'assessmentItemID', 'testId', 'lag_time', 'elapsed_time', 'item_acc', 'user_acc', 'answerCode']
    
    if scaling :
        time_col = ['lag_time', 'elapsed_time']
        total_time, test_time = total[time_col], test[time_col]

        rb_scaler = RobustScaler()
        total_time_scaled = pd.DataFrame(rb_scaler.fit_transform(total_time), columns = time_col)
        test_time_scaled = pd.DataFrame(rb_scaler.transform(test_time), columns = time_col)

        total_df = pd.concat([total.drop(columns = time_col), total_time_scaled], axis = 1)
        test_df = pd.concat([test.drop(columns = time_col), test_time_scaled], axis = 1)
    
    else : 
        total_df = total.copy()
        test_df = test.copy()
    
    # valid_indices = set(df[df['answerCode'] != -1].index).intersection(set(df.reset_index().groupby('userID', as_index = False).last().set_index('index').index))
    # train_df = df.loc[~df.index.isin(valid_indices)]
    # valid_df = df.loc[df.index.isin(valid_indices)]
    
    print(total_df.shape, total.shape)
    valid_size = 0.98
    train_df = total_df[:int(total_df.shape[0] * valid_size)]
    valid_df = total_df[int(total_df.shape[0] * valid_size):]
    print(f'Train : {train_df[features].shape}, Valid : {valid_df[features].shape}')
    print('=' * 50)

    grouping(cfg, train_df, features, 'Train_SP')
    grouping(cfg, valid_df, features, 'Valid_SP')
    grouping(cfg, test_df,  features, 'Test_SP')
    print('Finish Preprocess')


if __name__ == '__main__' :
    cfg = load_config('../config.yaml')

    total_df = pd.read_csv(cfg['data_dir'] + cfg['total_data_name'])
    test_df = pd.read_csv(cfg['data_dir'] + cfg['test_data_name'])
    Preprocess(cfg, total_df, test_df, cfg['scale'])