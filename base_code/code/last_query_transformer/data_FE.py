import numpy as np
import pandas as pd
from ....ML_Models.ELO import ELO

import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, help="")

    args = parser.parse_args()

    return args


def Feature_Engineering(df) :
    '''
    전처리 및 변수 생성을 위한 함수
    '''
    df.sort_values(by = ['userID', 'Timestamp'], inplace = True)

    # 시험지 대분류, 시험지 고유 번호, 시험 문항 번호
    df['testID'] = df['assessmentItemID'].apply(lambda x : x[1:7])
    df['testCode'] = df['assessmentItemID'].apply(lambda x : x[2:3]).astype('int8')
    df['testNum'] = df['assessmentItemID'].apply(lambda x : x[4:7]).astype('int16')
    df['problemID'] = df['assessmentItemID'].apply(lambda x : x[7:]).astype('int8')
    df['problemID_Norm'] = (df['problemID'] - min(df['problemID'])) / (max(df['problemID']) - min(df['problemID']))
    
    # testCode 별 중복되는 KnowledgeTag 전처리
    df.loc[df['KnowledgeTag'] == 7863, 'testCode'] = 7
    # df.loc[df['assessmentItemID'] == 'A080037007', 'KnowledgeTag'] = 4686
    
    # 문제를 푼 날짜 정보 중 년, 월, 일, 시간, 요일, 평일 여부 정보
    df['year'] = df['Timestamp'].dt.year.astype('int16')
    df['month'] = df['Timestamp'].dt.month.astype('int8')
    df['day'] = df['Timestamp'].dt.day.astype('int8')
    df['hour'] = df['Timestamp'].dt.hour.astype('int8')
    df['dow'] = df['Timestamp'].dt.dayofweek.astype('int8')
    df['date'] = df['Timestamp'].dt.date
    df['weekday'] = np.where(df['dow'] >= 5, 0, 1).astype('int8')
    
    # 문제를 푸는데 걸린 시간
    df['ElapsedTime'] = df.groupby(['userID', 'testID'], as_index = False)['Timestamp'].diff().shift(-1)
    df['ElapsedTime'] = df['ElapsedTime'].apply(lambda x : x.total_seconds())
    df.loc[df['ElapsedTime'] > 1200, 'ElapsedTime'] = np.nan # 20분(1200초) 초과 -> 결측
    df['ElapsedTime'] = df.groupby('problemID')['ElapsedTime'].transform(lambda x: x.fillna(x.mean())) # 문제 번호 별 평균 풀이 시간으로 결측치 대체
    
    # 유저가 문제를 푼 시간 이동평균
    df['ElapsedTime_Rolling2'] = df.groupby('userID')['ElapsedTime'].rolling(2).mean().values
    df['ElapsedTime_Rolling2'] = df['ElapsedTime_Rolling2'].fillna(0)
    df['ElapsedTime_Rolling3'] = df.groupby('userID')['ElapsedTime'].rolling(3).mean().values
    df['ElapsedTime_Rolling3'] = df['ElapsedTime_Rolling3'].fillna(0)
    df['ElapsedTime_Rolling4'] = df.groupby('userID')['ElapsedTime'].rolling(4).mean().values
    df['ElapsedTime_Rolling4'] = df['ElapsedTime_Rolling4'].fillna(0)
    df['ElapsedTime_Rolling5'] = df.groupby('userID')['ElapsedTime'].rolling(5).mean().values
    df['ElapsedTime_Rolling5'] = df['ElapsedTime_Rolling5'].fillna(0)
    
    # userID, assessmentItemID, testID, testCode, testNum, problemID, KnowledgeTag 별 평균 문제 풀이 시간
    df['user_ElaspedTime_avg'] = df['userID'].map(df.groupby('userID')['ElapsedTime'].mean().to_dict())
    df['item_ElaspedTime_avg'] = df['assessmentItemID'].map(df.groupby('assessmentItemID')['ElapsedTime'].mean().to_dict())
    df['testID_ElaspedTime_avg'] = df['testID'].map(df.groupby('testID')['ElapsedTime'].mean().to_dict())
    df['testCode_ElaspedTime_avg'] = df['testCode'].map(df.groupby('testCode')['ElapsedTime'].mean().to_dict())
    df['testNum_ElaspedTime_avg'] = df['testNum'].map(df.groupby('testNum')['ElapsedTime'].mean().to_dict())
    df['problemID_ElaspedTime_avg'] = df['problemID'].map(df.groupby('problemID')['ElapsedTime'].mean().to_dict())
    df['tag_ElaspedTime_avg'] = df['KnowledgeTag'].map(df.groupby('KnowledgeTag')['ElapsedTime'].mean().to_dict())
    
    # 유저가 문제를 실제로 풀었는지 여부 (9초부터 평균 정답률 이상이므로, 9초 미만은 찍은 걸로 간주)
    df['Real_Solved'] = np.where(df['ElapsedTime'] >= 9, 1, 0).astype('int8')

    # 해당 문항을 맞힌 or 틀린 학생의 평균 문제 풀이 소요시간
    df['Correct_User_ElapsedTime'] = df['assessmentItemID'].map(df[df['answerCode'] == 1].groupby('assessmentItemID')['ElapsedTime'].mean().to_dict())
    df['Wrong_User_ElapsedTime'] = df['assessmentItemID'].map(df[df['answerCode'] == 0].groupby('assessmentItemID')['ElapsedTime'].mean().to_dict())    

    
    # 유저가 정답을 맞힌 횟수 / 문제를 푼 횟수 / 정답률을 시간순으로 누적해서 계산
    df['user_sum'] = df.groupby('userID')['answerCode'].transform(lambda x : x.cumsum().shift(1)).fillna(0).astype('int16')
    df['user_cnt'] = df.groupby('userID')['answerCode'].cumcount().astype('int16')
    df['user_acc'] = (df['user_sum'] / df['user_cnt']).fillna(0)

    # 유저가 assessmentItemID 별 정답을 맞힌 횟수 / 문제를 푼 횟수 / 정답률을 시간순으로 누적해서 계산
    df['user_itemID_sum'] = df.groupby(['userID', 'assessmentItemID'])['answerCode'].transform(lambda x : x.cumsum().shift(1)).fillna(0).astype('int16')
    df['user_itemID_cnt'] = df.groupby(['userID', 'assessmentItemID'])['answerCode'].cumcount().astype('int16')
    df['user_itemID_acc'] = (df['user_itemID_sum'] / df['user_itemID_cnt']).fillna(0)
    
    # 유저가 testID 별 정답을 맞힌 횟수 / 문제를 푼 횟수 / 정답률을 시간순으로 누적해서 계산
    df['user_testID_sum'] = df.groupby(['userID', 'testID', 'date'])['answerCode'].transform(lambda x : x.cumsum().shift(1)).fillna(0).astype('int16')
    df['user_testID_cnt'] = df.groupby(['userID', 'testID', 'date'])['answerCode'].cumcount().astype('int16')
    df['user_testID_acc'] = (df['user_testID_sum'] / df['user_testID_cnt']).fillna(0)

    # 유저가 testID 별 정답을 맞힌 횟수 / 문제를 푼 횟수 / 정답률을 시간순으로 누적해서 계산
    df['user_testCode_sum'] = df.groupby(['userID', 'testCode'])['answerCode'].transform(lambda x : x.cumsum().shift(1)).fillna(0).astype('int16')
    df['user_testCode_cnt'] = df.groupby(['userID', 'testCode'])['answerCode'].cumcount().astype('int16')
    df['user_testCode_acc'] = (df['user_testCode_sum'] / df['user_testCode_cnt']).fillna(0)

    # 유저가 testNum 별 정답을 맞힌 횟수 / 문제를 푼 횟수 / 정답률을 시간순으로 누적해서 계산
    df['user_testNum_sum'] = df.groupby(['userID', 'testNum'])['answerCode'].transform(lambda x : x.cumsum().shift(1)).fillna(0).astype('int16')
    df['user_testNum_cnt'] = df.groupby(['userID', 'testNum'])['answerCode'].cumcount().astype('int16')
    df['user_testNum_acc'] = (df['user_testNum_sum'] / df['user_testNum_cnt']).fillna(0)
    
    # 유저가 problemID 별 정답을 맞힌 횟수 / 문제를 푼 횟수 / 정답률을 시간순으로 누적해서 계산
    df['user_problemID_sum'] = df.groupby(['userID', 'problemID'])['answerCode'].transform(lambda x : x.cumsum().shift(1)).fillna(0).astype('int16')
    df['user_problemID_cnt'] = df.groupby(['userID', 'problemID'])['answerCode'].cumcount().astype('int16')
    df['user_problemID_acc'] = (df['user_problemID_sum'] / df['user_problemID_cnt']).fillna(0)
    
    # 유저가 KnowledgeTag 별 정답을 맞힌 횟수 / 문제를 푼 횟수 / 정답률을 시간순으로 누적해서 계산
    df['user_tag_sum'] = df.groupby(['userID', 'KnowledgeTag'])['answerCode'].transform(lambda x : x.cumsum().shift(1)).fillna(0).astype('int16')
    df['user_tag_cnt'] = df.groupby(['userID', 'KnowledgeTag'])['answerCode'].cumcount().astype('int16')
    df['user_tag_acc'] = (df['user_tag_sum'] / df['user_tag_cnt']).fillna(0)
    
    
    # 전체 유저의 assessmentItemID, testID, testCode, testNum, ProblemID, KnowledgeTag의 정답률 / 정답을 맞힌 횟수 / 문제를 푼 횟수 계산
    correct_i = df.groupby(['assessmentItemID'])['answerCode'].agg(['sum', 'count', 'mean'])
    correct_i.columns = ['itemID_sum', 'itemID_cnt', 'itemID_acc']
    correct_ti = df.groupby(['testID'])['answerCode'].agg(['sum', 'count', 'mean'])
    correct_ti.columns = ['testID_sum', 'testID_cnt', 'testID_acc']
    correct_tc = df.groupby(['testCode'])['answerCode'].agg(['sum', 'count', 'mean'])
    correct_tc.columns = ['testCode_sum', 'testCode_cnt', 'testCode_acc']
    correct_tn = df.groupby(['testNum'])['answerCode'].agg(['sum', 'count', 'mean'])
    correct_tn.columns = ['testNum_sum', 'testNum_cnt', 'testNum_acc']
    correct_p = df.groupby(['problemID'])['answerCode'].agg(['sum', 'count', 'mean'])
    correct_p.columns = ['problemID_sum', 'problemID_cnt', 'problemID_acc']
    correct_k = df.groupby(['KnowledgeTag'])['answerCode'].agg(['sum', 'count', 'mean'])
    correct_k.columns = ['tag_sum', 'tag_cnt', 'tag_acc']

    df = pd.merge(df, correct_i, on = ['assessmentItemID'], how = 'left')
    df = pd.merge(df, correct_ti, on = ['testID'], how = 'left')
    df = pd.merge(df, correct_tc, on = ['testCode'], how = 'left')
    df = pd.merge(df, correct_tn, on = ['testNum'], how = 'left')
    df = pd.merge(df, correct_p, on = ['problemID'], how = 'left')
    df = pd.merge(df, correct_k, on = ['KnowledgeTag'], how = 'left')
    
    convert_dtype_col = ['itemID_sum', 'itemID_cnt', 'testID_sum', 'testID_cnt', 'testCode_sum', 'testCode_cnt',
                         'testNum_sum', 'testNum_cnt', 'problemID_sum', 'problemID_cnt', 'tag_sum', 'tag_cnt']
    df[convert_dtype_col] = df[convert_dtype_col].astype('int32')
    
    # assessmentItemID, testID, KnowledgeTag의 유저별 정답률, 정답 수, 문제 풀이 수를 계산
    # mean : 정답률, sum : 정답 수, count : 문제 풀이 수
    # correct_user_i = df.groupby(['userID', 'assessmentItemID'])['answerCode'].agg(['mean', 'sum', 'count'])
    # correct_user_i.columns = ['user_itemID_mean', 'user_item_sum', 'user_item_count']
    # correct_user_t = df.groupby(['userID', 'testID'])['answerCode'].agg(['mean', 'sum', 'count'])
    # correct_user_t.columns = ['user_test_mean', 'user_test_sum', 'user_test_count']
    # correct_user_k = df.groupby(['userID', 'KnowledgeTag'])['answerCode'].agg(['mean', 'sum', 'count'])
    # correct_user_k.columns = ['user_tag_mean', 'user_tag_sum', 'user_tag_count']
    # 
    # df = pd.merge(df, correct_user_i, on = ['userID', 'assessmentItemID'], how = 'left')
    # df = pd.merge(df, correct_user_t, on = ['userID', 'testID'], how = 'left')
    # df = pd.merge(df, correct_user_k, on = ['userID', 'KnowledgeTag'], how = 'left')

    # assessmentItemID, testID, testCode, testNum, KnowledgeTag가 평균 이상으로 노출되었는지 여부
    df['itemID_high_freq'] = np.where(df['itemID_cnt'] >= df['assessmentItemID'].value_counts().mean(), 1, 0).astype('int8')
    df['testID_high_freq'] = np.where(df['testID_cnt'] >= df['testID'].value_counts().mean(), 1, 0).astype('int8')
    df['testCode_high_freq'] = np.where(df['testCode_cnt'] >= df['testCode'].value_counts().mean(), 1, 0).astype('int8')
    df['testNum_high_freq'] = np.where(df['testNum_cnt'] >= df['testNum'].value_counts().mean(), 1, 0).astype('int8')
    df['problemID_high_freq'] = np.where(df['problemID_cnt'] >= df['problemID'].value_counts().mean(), 1, 0).astype('int8')
    df['tag_high_freq'] = np.where(df['tag_cnt'] >= df['KnowledgeTag'].value_counts().mean(), 1, 0).astype('int8')


    # 유저가 해당 문제를 이전에 몇 번 풀었는지
    df['user_past'] = df.groupby(['userID', 'assessmentItemID'])['answerCode'].shift(1).fillna(0).astype('int8')
    df['user_past_solved'] = df.groupby(['userID', 'assessmentItemID'])['user_past'].cumsum().astype('int8')

    # 상대적인 정답률
    df['relative_correct_rate'] = df['answerCode'] - df['itemID_acc']
    
    # 이전 문제의 정답 여부, 정답률, 상대적인 정답률 (첫 문항은 결측치인데, 대부분 첫 문제는 맞추니까 결측치를 1로 대체)
    df['is_correct_before1'] = df.groupby(['userID', 'testID'], as_index = False)['answerCode'].shift(1).fillna(1).astype('int8')
    df['correct_rate_before1'] = df.groupby(['userID', 'testID'], as_index = False)['itemID_acc'].shift(1).fillna(1)
    df['relative_correct_rate_before1'] = df.groupby(['userID', 'testID'], as_index = False)['relative_correct_rate'].shift(1).fillna(1)

    df['is_correct_before2'] = df.groupby(['userID', 'testID'], as_index = False)['answerCode'].shift(2).fillna(1).astype('int8')
    df['correct_rate_before2'] = df.groupby(['userID', 'testID'], as_index = False)['itemID_acc'].shift(2).fillna(1)
    df['relative_correct_rate_before2'] = df.groupby(['userID', 'testID'], as_index = False)['relative_correct_rate'].shift(2).fillna(1)
    
    df['is_correct_before3'] = df.groupby(['userID', 'testID'], as_index = False)['answerCode'].shift(3).fillna(1).astype('int8')
    df['correct_rate_before3'] = df.groupby(['userID', 'testID'], as_index = False)['itemID_acc'].shift(3).fillna(1)
    df['relative_correct_rate_before3'] = df.groupby(['userID', 'testID'], as_index = False)['relative_correct_rate'].shift(3).fillna(1)

    df['is_correct_before4'] = df.groupby(['userID', 'testID'], as_index = False)['answerCode'].shift(4).fillna(1).astype('int8')
    df['correct_rate_before4'] = df.groupby(['userID', 'testID'], as_index = False)['itemID_acc'].shift(4).fillna(1)
    df['relative_correct_rate_before4'] = df.groupby(['userID', 'testID'], as_index = False)['relative_correct_rate'].shift(4).fillna(1)
    
    df['is_correct_before5'] = df.groupby(['userID', 'testID'], as_index = False)['answerCode'].shift(5).fillna(1).astype('int8')
    df['correct_rate_before5'] = df.groupby(['userID', 'testID'], as_index = False)['itemID_acc'].shift(5).fillna(1)
    df['relative_correct_rate_before5'] = df.groupby(['userID', 'testID'], as_index = False)['relative_correct_rate'].shift(5).fillna(1)
    
    
    # ELO Rating (theta, beta estimate)
    theta_estimate, beta_estimate = ELO(df)
    df['theta'] = df['userID'].astype(str).map(theta_estimate)
    df['beta'] = df['assessmentItemID'].map(beta_estimate)
    
    # 사용하지 않는 변수 제거
    df.drop(columns = ['testId', 'Timestamp', 'date', 'user_past'], inplace = True)
    return df


def main():
    args = parse_args()

    dtype = {'userID' : 'int16',
         'answerCode' : 'int8',
         'KnowledgeTag' : 'int16'}   

    DATA_PATH = args.data_dir

    train = pd.read_csv(DATA_PATH + 'train_data.csv', dtype = dtype, parse_dates = ['Timestamp'])
    train = train.sort_values(by = ['userID', 'Timestamp']).reset_index(drop = True)

    test = pd.read_csv(DATA_PATH + 'test_data.csv', dtype = dtype, parse_dates = ['Timestamp'])
    test = test.sort_values(by = ['userID', 'Timestamp']).reset_index(drop = True)

    ### train data 전처리
    data = Feature_Engineering(train.copy())
    data.to_csv(DATA_PATH + 'train_data.csv', index=False)

    ### test data 전처리
    data = pd.concat([train, test], axis = 0).reset_index(drop = True)
    data = Feature_Engineering(data)

    data = data.iloc[train.index.stop:]
    data.to_csv(DATA_PATH + 'test_data.csv', index=False)


if __name__ == '__main__':
    main()

