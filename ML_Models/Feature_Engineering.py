import pandas as pd
import numpy as np
from ELO import ELO

class FeatureEngineering :
    def __init__(self, df) :
        pass
    
    ### 시험지 관련 변수
    def Feature_UserID(self, df) :
        '''
        `userID` 변수
        '''
        userID = df['userID']
        return userID.values
    
    def Feature_ItemID(self, df) :
        '''
        `assessmentItemID` 변수
        '''
        itemID = df['assessmentItemID']
        return itemID.values

    def Feature_answerCode(self, df) :
        '''
        `Feature_answerCode` 변수
        '''
        answerCode = df['answerCode']
        return answerCode.values
    
    def Feature_Tag(self, df) :
        '''
        KnowledgeTag의 이상치 처리
        '''
        # df.loc[df['KnowledgeTag'] == 7863, 'testCode'] = 7
        df.loc[df['assessmentItemID'] == 'A080037007', 'KnowledgeTag'] = 4686
        KnowledgeTag = df['KnowledgeTag']
        return KnowledgeTag.values        
    
    def Feature_testID(self, df) :
        '''
        testID의 필요 부분만 추출 (`assessmentItemID`의 2~7번째 자리)
        '''
        testID = df['assessmentItemID'].apply(lambda x : x[1:7])
        return testID.values
    
    def Feature_testCode(self, df) :
        '''
        시험지의 대분류 (`assessmentItemID`의 3번째 자리)
        '''
        testCode = df['assessmentItemID'].apply(lambda x : x[2:3]).astype('int8')
        return testCode.values

    def Feature_testNum(self, df) :
        '''
        시험지의 대분류 (`assessmentItemID`의 4~6번째 자리)
        '''
        testNum = df['assessmentItemID'].apply(lambda x : x[4:7]).astype('int32')
        return testNum.values

    def Feature_problemID(self, df) :
        '''
        시험지의 문항 번호 (`assessmentItemID`의 8~10번째 자리)
        '''
        problemID = df['assessmentItemID'].apply(lambda x : x[7:]).astype('int8')
        return problemID.values
    
    def Feature_problemID_Norm(self, df) :
        '''
        시험지의 문항 번호 Min-Max Scaling (`assessmentItemID`의 8~10번째 자리)
        '''
        problemID = df['assessmentItemID'].apply(lambda x : x[7:]).astype('int8')
        problemID_Norm = (problemID - min(problemID)) / (max(problemID) - min(problemID))
        return problemID_Norm.values

    def Feature_Total_Problem(self, df) :
        '''
        시험지 별 문항의 총 개수
        '''
        df['testID'] = self.Feature_testID(df)
        df['problemID'] = self.Feature_problemID(df)
        total_problem = df['testID'].map(df.groupby('testID')['problemID'].max().to_dict()).astype('int8')
        return total_problem.values

    ### Timestamp 관련 변수 생성
    
    def Feature_year(self, df) :
        '''
        문제를 푼 날짜 정보 중 년
        '''
        year = df['Timestamp'].dt.year.astype('int32')
        return year.values
    
    def Feature_quarter(self, df) :
        '''
        문제를 푼 날짜 정보 중 분기
        '''
        quarter = df['Timestamp'].dt.quarter.astype('int8')
        return quarter.values

    def Feature_month(self, df) :
        '''
        문제를 푼 날짜 정보 중 월
        '''
        month = df['Timestamp'].dt.month.astype('int8')
        return month.values

    def Feature_day(self, df) :
        '''
        문제를 푼 날짜 정보 중 일
        '''
        day = df['Timestamp'].dt.day.astype('int8')
        return day.values

    def Feature_hour(self, df) :
        '''
        문제를 푼 날짜 정보 중 시간
        '''
        hour = df['Timestamp'].dt.hour.astype('int8')
        return hour.values

    def Feature_dow(self, df) :
        '''
        문제를 푼 날짜 정보 중 요일
        '''
        dow = df['Timestamp'].dt.dayofweek.astype('int8')
        return dow.values

    def Feature_weekday(self, df) :
        '''
        문제를 푼 날짜 정보 중 평일 여부 (0 : 주말, 1 : 평일)
        '''
        dow = self.Feature_dow(df)
        weekday = np.where(dow >= 5, 0, 1).astype('int8')
        return weekday

    def Feature_LagTime(self, df) :
        time_dict = {}
        lag_time = np.zeros(len(df), dtype = np.float32)
        for idx, (user, time, test) in enumerate(df[['userID', 'Timestamp', 'testID']].values) :

            if user not in time_dict :
                # 문제를 처음 푸는 유저
                lag_time[idx] = 0
                time_dict[user] = [time, test, 0] # 마지막 Timestamp, 마지막 testID, 마지막 LagTime(= 0)

            else :
                # 해당 시험지를 풀었다면
                if test == time_dict[user][1] :
                    lag_time[idx] = time_dict[user][2] # 마지막 LagTime 입력

                # 처음 보는 시험지라면
                else :
                    lag_time[idx] = (time - time_dict[user][0]).total_seconds()
                    time_dict[user][0] = time
                    time_dict[user][1] = test
                    time_dict[user][2] = lag_time[idx]
        LagTime = lag_time / 1000 / 60 # 분 단위로 변환
        LagTime = LagTime.clip(0, 1440)
        return LagTime
    
    def Feature_ElapsedTime(self, df) :
        '''
        유저가 문제를 푸는데 걸린 시간
        (3600초 이상 결측치 처리 후 문제 번호(problemID) 별 ElapsedTime의 평균값으로 대체)
        '''
        df['testID'] = self.Feature_testID(df)
        ElaspedTime = df.groupby(['userID', 'testID'], as_index = False)['Timestamp'].diff().shift(0).dt.total_seconds() # shift(-1)
        ElaspedTime = np.nan_to_num(ElaspedTime, nan = 0) # 유저별 첫번째 문항의 풀이 시간 -> 0
        ElaspedTime = np.where(ElaspedTime > 3600, np.nan, ElaspedTime) # 1시간 이상 결측치 처리
        
        df['problemID'] = self.Feature_problemID(df)
        df['ElaspedTime'] = ElaspedTime
        
        ElaspedTime_Filled = df.groupby('problemID')['ElaspedTime'].transform(lambda x : x.fillna(x.mean()))
        return ElaspedTime_Filled.values

    def Feature_ElapsedTime_Rolling_Average2(self, df) :
        '''
        유저가 문제를 푸는데 걸린 시간 이동평균 (Rolling = 2)
        '''
        df['ElapsedTime'] = self.Feature_ElapsedTime(df)
        df['ElapsedTime_Rolling2'] = df.groupby('userID')['ElapsedTime'].rolling(2).mean().values
        ElapsedTime_Rolling2 =  df['ElapsedTime_Rolling2'].fillna(0)
        return ElapsedTime_Rolling2.values
    
    def Feature_ElapsedTime_Rolling_Average3(self, df) :
        '''
        유저가 문제를 푸는데 걸린 시간 이동평균 (Rolling = 3)
        '''
        df['ElapsedTime'] = self.Feature_ElapsedTime(df)
        df['ElapsedTime_Rolling3'] = df.groupby('userID')['ElapsedTime'].rolling(3).mean().values
        ElapsedTime_Rolling3 =  df['ElapsedTime_Rolling3'].fillna(0)
        return ElapsedTime_Rolling3.values
    
    def Feature_ElapsedTime_Rolling_Average4(self, df) :
        '''
        유저가 문제를 푸는데 걸린 시간 이동평균 (Rolling = 4)
        '''
        df['ElapsedTime'] = self.Feature_ElapsedTime(df)
        df['ElapsedTime_Rolling4'] = df.groupby('userID')['ElapsedTime'].rolling(4).mean().values
        ElapsedTime_Rolling4 =  df['ElapsedTime_Rolling4'].fillna(0)
        return ElapsedTime_Rolling4.values
    
    def Feature_ElapsedTime_Rolling_Average5(self, df) :
        '''
        유저가 문제를 푸는데 걸린 시간 이동평균 (Rolling = 5)
        '''
        df['ElapsedTime'] = self.Feature_ElapsedTime(df)
        df['ElapsedTime_Rolling5'] = df.groupby('userID')['ElapsedTime'].rolling(5).mean().values
        ElapsedTime_Rolling5 =  df['ElapsedTime_Rolling5'].fillna(0)
        return ElapsedTime_Rolling5.values

    def Feature_User_ElapsedTime_Average(self, df) :
        '''
        `userID` 별 문제를 푸는데 걸린 시간의 평균
        '''
        df['ElapsedTime'] = self.Feature_ElapsedTime(df)
        user_ElaspedTime_avg = df['userID'].map(df.groupby('userID')['ElapsedTime'].mean().to_dict())
        return user_ElaspedTime_avg.values

    def Feature_Item_ElapsedTime_Average(self, df) :
        '''
        `assessmentItemID` 별 문제를 푸는데 걸린 시간의 평균
        '''
        df['ElapsedTime'] = self.Feature_ElapsedTime(df)
        item_ElaspedTime_avg = df['assessmentItemID'].map(df.groupby('assessmentItemID')['ElapsedTime'].mean().to_dict())
        return item_ElaspedTime_avg.values

    def Feature_testID_ElapsedTime_Average(self, df) :
        '''
        `testID` 별 문제를 푸는데 걸린 시간의 평균
        '''
        df['testID'] = self.Feature_testID(df)
        df['ElapsedTime'] = self.Feature_ElapsedTime(df)
        testID_ElaspedTime_avg = df['testID'].map(df.groupby('testID')['ElapsedTime'].mean().to_dict())
        return testID_ElaspedTime_avg.values

    def Feature_testCode_ElapsedTime_Average(self, df) :
        '''
        `testCode` 별 문제를 푸는데 걸린 시간의 평균
        '''
        df['testCode'] = self.Feature_testCode(df)
        df['ElapsedTime'] = self.Feature_ElapsedTime(df)
        testCode_ElaspedTime_avg = df['testCode'].map(df.groupby('testCode')['ElapsedTime'].mean().to_dict())
        return testCode_ElaspedTime_avg.values

    def Feature_testNum_ElapsedTime_Average(self, df) :
        '''
        `testNum` 별 문제를 푸는데 걸린 시간의 평균
        '''
        df['testNum'] = self.Feature_testNum(df)
        df['ElapsedTime'] = self.Feature_ElapsedTime(df)
        testNum_ElaspedTime_avg = df['testNum'].map(df.groupby('testNum')['ElapsedTime'].mean().to_dict())
        return testNum_ElaspedTime_avg.values

    def Feature_problemID_ElapsedTime_Average(self, df) :
        '''
        `problemID` 별 문제를 푸는데 걸린 시간의 평균
        '''
        df['problemID'] = self.Feature_problemID(df)
        df['ElapsedTime'] = self.Feature_ElapsedTime(df)
        problemID_ElaspedTime_avg = df['problemID'].map(df.groupby('problemID')['ElapsedTime'].mean().to_dict())
        return problemID_ElaspedTime_avg.values

    def Feature_Tag_ElapsedTime_Average(self, df) :
        '''
        `KnowledgeTag` 별 문제를 푸는데 걸린 시간의 평균
        '''
        df['ElapsedTime'] = self.Feature_ElapsedTime(df)
        tag_ElaspedTime_avg = df['KnowledgeTag'].map(df.groupby('KnowledgeTag')['ElapsedTime'].mean().to_dict())
        return tag_ElaspedTime_avg.values

    def Feature_Real_Solved(self, df) :
        '''
        유저가 문제를 실제로 풀었는지 여부
        (9초부터 평균 정답률(약 65%) 이상이므로, 9초 미만은 찍은 걸로 간주)
        '''
        ElapsedTime = self.Feature_ElapsedTime(df)
        Real_Solved = np.where(ElapsedTime >= 9, 1, 0).astype('int8')
        return Real_Solved

    def Feature_Correct_ElapsedTime_Average(self, df) :
        '''
        해당 문제를 맞힌 학생의 평균 문제 풀이 시간
        '''
        df['ElapsedTime'] = self.Feature_ElapsedTime(df)
        Correct_ElapsedTime = df['assessmentItemID'].map(df[df['answerCode'] == 1].groupby('assessmentItemID')['ElapsedTime'].mean().to_dict())
        return Correct_ElapsedTime.values

    def Feature_Wrong_ElapsedTime_Average(self, df) :
        '''
        해당 문제를 틀린 학생의 평균 문제 풀이 시간
        '''
        df['ElapsedTime'] = self.Feature_ElapsedTime(df)
        Wrong_ElapsedTime = df['assessmentItemID'].map(df[df['answerCode'] == 0].groupby('assessmentItemID')['ElapsedTime'].mean().to_dict())
        return Wrong_ElapsedTime.values

    ### 유저 별로 여러 변수 별 정답 수 / 문제 풀이 수 / 정답률 계산 (시간 순으로 누적)
    
    def Feature_User_Sum(self, df) :
        '''
        유저 별 정답을 맞힌 횟수 (시간 순으로 누적해서 계산)
        '''
        User_Sum = df.groupby('userID')['answerCode'].transform(lambda x : x.cumsum().shift(1)).fillna(0).astype('int32')
        return User_Sum.values

    def Feature_User_Count(self, df) :
        '''
        유저 별 문제를 푼 횟수 (시간 순으로 누적해서 계산)
        '''
        User_Count = df.groupby('userID')['answerCode'].cumcount().astype('int32')
        return User_Count.values
    
    def Feature_User_Acc(self, df) :
        '''
        유저 별 정답률 (시간 순으로 누적해서 계산)
        '''
        User_Acc = self.Feature_User_Sum(df) / self.Feature_User_Count(df)
        User_Acc = np.nan_to_num(User_Acc, nan = 0)
        return User_Acc
    
    def Feature_User_Item_Sum(self, df) :
        '''
        유저의 `assessmentItemID` 별 정답을 맞힌 횟수 (시간 순으로 누적해서 계산)
        '''
        User_Item_Sum = df.groupby(['userID', 'assessmentItemID'])['answerCode'].transform(lambda x : x.cumsum().shift(1)).fillna(0).astype('int32')
        return User_Item_Sum.values

    def Feature_User_Item_Count(self, df) :
        '''
        유저의 `assessmentItemID` 별 문제를 푼 횟수 (시간 순으로 누적해서 계산)
        '''
        User_Item_Count = df.groupby(['userID', 'assessmentItemID'])['answerCode'].cumcount().astype('int32')
        return User_Item_Count.values
    
    def Feature_User_Item_Acc(self, df) :
        '''
        유저의 `assessmentItemID` 별 정답률 (시간 순으로 누적해서 계산)
        '''
        User_Item_Acc = self.Feature_User_Item_Sum(df) / self.Feature_User_Item_Count(df)
        User_Item_Acc = np.nan_to_num(User_Item_Acc, nan = 0)
        return User_Item_Acc

    def Feature_User_testID_Sum(self, df) :
        '''
        유저의 `testID` 별 정답을 맞힌 횟수 (시간 순으로 누적해서 계산)
        '''
        df['testID'] = self.Feature_testID(df)
        User_testID_Sum = df.groupby(['userID', 'testID'])['answerCode'].transform(lambda x : x.cumsum().shift(1)).fillna(0).astype('int32')
        return User_testID_Sum.values

    def Feature_User_testID_Count(self, df) :
        '''
        유저의 `testID` 별 문제를 푼 횟수 (시간 순으로 누적해서 계산)
        '''
        df['testID'] = self.Feature_testID(df)
        User_testID_Count = df.groupby(['userID', 'testID'])['answerCode'].cumcount().astype('int32')
        return User_testID_Count.values
    
    def Feature_User_testID_Acc(self, df) :
        '''
        유저의 `testID` 별 정답률 (시간 순으로 누적해서 계산)
        '''
        User_testID_Acc = self.Feature_User_testID_Sum(df) / self.Feature_User_testID_Count(df)
        User_testID_Acc = np.nan_to_num(User_testID_Acc, nan = 0)
        return User_testID_Acc

    def Feature_User_testCode_Sum(self, df) :
        '''
        유저의 `testCode` 별 정답을 맞힌 횟수 (시간 순으로 누적해서 계산)
        '''
        df['testCode'] = self.Feature_testCode(df)
        User_testCode_Sum = df.groupby(['userID', 'testCode'])['answerCode'].transform(lambda x : x.cumsum().shift(1)).fillna(0).astype('int32')
        return User_testCode_Sum.values

    def Feature_User_testCode_Count(self, df) :
        '''
        유저의 `testCode` 별 문제를 푼 횟수 (시간 순으로 누적해서 계산)
        '''
        df['testCode'] = self.Feature_testCode(df)
        User_testCode_Count = df.groupby(['userID', 'testCode'])['answerCode'].cumcount().astype('int32')
        return User_testCode_Count.values
    
    def Feature_User_testCode_Acc(self, df) :
        '''
        유저의 `testCode` 별 정답률 (시간 순으로 누적해서 계산)
        '''
        User_testCode_Acc = self.Feature_User_testCode_Sum(df) / self.Feature_User_testCode_Count(df)
        User_testCode_Acc = np.nan_to_num(User_testCode_Acc, nan = 0)
        return User_testCode_Acc

    def Feature_User_testNum_Sum(self, df) :
        '''
        유저의 `testNum` 별 정답을 맞힌 횟수 (시간 순으로 누적해서 계산)
        '''
        df['testNum'] = self.Feature_testNum(df)
        User_testNum_Sum = df.groupby(['userID', 'testNum'])['answerCode'].transform(lambda x : x.cumsum().shift(1)).fillna(0).astype('int32')
        return User_testNum_Sum.values

    def Feature_User_testNum_Count(self, df) :
        '''
        유저의 `testNum` 별 문제를 푼 횟수 (시간 순으로 누적해서 계산)
        '''
        df['testNum'] = self.Feature_testNum(df)
        User_testNum_Count = df.groupby(['userID', 'testNum'])['answerCode'].cumcount().astype('int32')
        return User_testNum_Count.values
    
    def Feature_User_testNum_Acc(self, df) :
        '''
        유저의 `testNum` 별 정답률 (시간 순으로 누적해서 계산)
        '''
        User_testNum_Acc = self.Feature_User_testNum_Sum(df) / self.Feature_User_testNum_Count(df)
        User_testNum_Acc = np.nan_to_num(User_testNum_Acc, nan = 0)
        return User_testNum_Acc

    def Feature_User_problemID_Sum(self, df) :
        '''
        유저의 `problemID` 별 정답을 맞힌 횟수 (시간 순으로 누적해서 계산)
        '''
        df['problemID'] = self.Feature_problemID(df)
        User_problemID_Sum = df.groupby(['userID', 'problemID'])['answerCode'].transform(lambda x : x.cumsum().shift(1)).fillna(0).astype('int32')
        return User_problemID_Sum.values

    def Feature_User_problemID_Count(self, df) :
        '''
        유저의 `problemID` 별 문제를 푼 횟수 (시간 순으로 누적해서 계산)
        '''
        df['problemID'] = self.Feature_problemID(df)
        User_problemID_Count = df.groupby(['userID', 'problemID'])['answerCode'].cumcount().astype('int32')
        return User_problemID_Count.values
    
    def Feature_User_problemID_Acc(self, df) :
        '''
        유저의 `problemID` 별 정답률 (시간 순으로 누적해서 계산)
        '''
        User_problemID_Acc = self.Feature_User_problemID_Sum(df) / self.Feature_User_problemID_Count(df)
        User_problemID_Acc = np.nan_to_num(User_problemID_Acc, nan = 0)
        return User_problemID_Acc

    def Feature_User_Tag_Sum(self, df) :
        '''
        유저의 `KnowledgeTag` 별 정답을 맞힌 횟수 (시간 순으로 누적해서 계산)
        '''
        User_Tag_Sum = df.groupby(['userID', 'KnowledgeTag'])['answerCode'].transform(lambda x : x.cumsum().shift(1)).fillna(0).astype('int32')
        return User_Tag_Sum.values

    def Feature_User_Tag_Count(self, df) :
        '''
        유저의 `KnowledgeTag` 별 문제를 푼 횟수 (시간 순으로 누적해서 계산)
        '''
        User_Tag_Count = df.groupby(['userID', 'KnowledgeTag'])['answerCode'].cumcount().astype('int32')
        return User_Tag_Count.values
    
    def Feature_User_Tag_Acc(self, df) :
        '''
        유저의 `KnowledgeTag` 별 정답률 (시간 순으로 누적해서 계산)
        '''
        User_Tag_Acc = self.Feature_User_Tag_Sum(df) / self.Feature_User_Tag_Count(df)
        User_Tag_Acc = np.nan_to_num(User_Tag_Acc, nan = 0)
        return User_Tag_Acc
    
    ### 전체 유저에 대해 여러 변수 별 정답 수 / 문제 풀이 수 / 정답률 계산
    
    def Feature_Item_Sum(self, df) :
        '''
        전체 유저의 `assessmentItemID` 별 정답을 맞힌 횟수
        '''
        Item_Sum = df['assessmentItemID'].map(df.groupby(['assessmentItemID'])['answerCode'].sum().astype('int32').to_dict())
        return Item_Sum.values

    def Feature_Item_Count(self, df) :
        '''
        전체 유저의 `assessmentItemID` 별 문제를 푼 횟수
        '''
        Item_Count = df['assessmentItemID'].map(df.groupby(['assessmentItemID'])['answerCode'].count().astype('int32').to_dict())
        return Item_Count.values
    
    def Feature_Item_Acc(self, df) :
        '''
        전체 유저의 `assessmentItemID` 별 정답률
        '''
        Item_Acc = self.Feature_Item_Sum(df) / self.Feature_Item_Count(df)
        Item_Acc = np.nan_to_num(Item_Acc, nan = 0)
        return Item_Acc

    def Feature_testID_Sum(self, df) :
        '''
        전체 유저의 `testID` 별 정답을 맞힌 횟수
        '''
        testID_Sum = df['testID'].map(df.groupby(['testID'])['answerCode'].sum().astype('int32').to_dict())
        return testID_Sum.values

    def Feature_testID_Count(self, df) :
        '''
        전체 유저의 `testID` 별 문제를 푼 횟수
        '''
        testID_Count = df['testID'].map(df.groupby(['testID'])['answerCode'].count().astype('int32').to_dict())
        return testID_Count.values
    
    def Feature_testID_Acc(self, df) :
        '''
        전체 유저의 `testID` 별 정답률
        '''
        testID_Acc = self.Feature_testID_Sum(df) / self.Feature_testID_Count(df)
        testID_Acc = np.nan_to_num(testID_Acc, nan = 0)
        return testID_Acc

    def Feature_testCode_Sum(self, df) :
        '''
        전체 유저의 `testCode` 별 정답을 맞힌 횟수
        '''
        testCode_Sum = df['testCode'].map(df.groupby(['testCode'])['answerCode'].sum().astype('int32').to_dict())
        return testCode_Sum.values

    def Feature_testCode_Count(self, df) :
        '''
        전체 유저의 `testCode` 별 문제를 푼 횟수
        '''
        testCode_Count = df['testCode'].map(df.groupby(['testCode'])['answerCode'].count().astype('int32').to_dict())
        return testCode_Count.values
    
    def Feature_testCode_Acc(self, df) :
        '''
        전체 유저의 `testCode` 별 정답률
        '''
        testCode_Acc = self.Feature_testCode_Sum(df) / self.Feature_testCode_Count(df)
        testCode_Acc = np.nan_to_num(testCode_Acc, nan = 0)
        return testCode_Acc

    def Feature_testNum_Sum(self, df) :
        '''
        전체 유저의 `testNum` 별 정답을 맞힌 횟수
        '''
        testNum_Sum = df['testNum'].map(df.groupby(['testNum'])['answerCode'].sum().astype('int32').to_dict())
        return testNum_Sum.values

    def Feature_testNum_Count(self, df) :
        '''
        전체 유저의 `testNum` 별 문제를 푼 횟수
        '''
        testNum_Count = df['testNum'].map(df.groupby(['testNum'])['answerCode'].count().astype('int32').to_dict())
        return testNum_Count.values
    
    def Feature_testNum_Acc(self, df) :
        '''
        전체 유저의 `testNum` 별 정답률
        '''
        testNum_Acc = self.Feature_testNum_Sum(df) / self.Feature_testNum_Count(df)
        testNum_Acc = np.nan_to_num(testNum_Acc, nan = 0)
        return testNum_Acc
    
    def Feature_problemID_Sum(self, df) :
        '''
        전체 유저의 `problemID` 별 정답을 맞힌 횟수
        '''
        problemID_Sum = df['problemID'].map(df.groupby(['problemID'])['answerCode'].sum().astype('int32').to_dict())
        return problemID_Sum.values

    def Feature_problemID_Count(self, df) :
        '''
        전체 유저의 `problemID` 별 문제를 푼 횟수
        '''
        problemID_Count = df['problemID'].map(df.groupby(['problemID'])['answerCode'].count().astype('int32').to_dict())
        return problemID_Count.values
    
    def Feature_problemID_Acc(self, df) :
        '''
        전체 유저의 `problemID` 별 정답률
        '''
        problemID_Acc = self.Feature_problemID_Sum(df) / self.Feature_problemID_Count(df)
        problemID_Acc = np.nan_to_num(problemID_Acc, nan = 0)
        return problemID_Acc

    def Feature_Tag_Sum(self, df) :
        '''
        전체 유저의 `KnowledgeTag` 별 정답을 맞힌 횟수
        '''
        Tag_Sum = df['KnowledgeTag'].map(df.groupby(['KnowledgeTag'])['answerCode'].sum().astype('int32').to_dict())
        return Tag_Sum.values

    def Feature_Tag_Count(self, df) :
        '''
        전체 유저의 `KnowledgeTag` 별 문제를 푼 횟수
        '''
        Tag_Count = df['KnowledgeTag'].map(df.groupby(['KnowledgeTag'])['answerCode'].count().astype('int32').to_dict())
        return Tag_Count.values
    
    def Feature_Tag_Acc(self, df) :
        '''
        전체 유저의 `KnowledgeTag` 별 정답률
        '''
        Tag_Acc = self.Feature_Tag_Sum(df) / self.Feature_Tag_Count(df)
        Tag_Acc = np.nan_to_num(Tag_Acc, nan = 0)
        return Tag_Acc
    
    def Feature_Item_High_Freq(self, df) :
        '''
        전체 유저에 대해 `assessmentItemID`가 평균 이상으로 노출되었는지 여부
        '''
        df['itemID_cnt'] = self.Feature_Item_Count(df)
        Item_High_Freq = np.where(df['itemID_cnt'] >= df['assessmentItemID'].value_counts().mean(), 1, 0).astype('int8')
        return Item_High_Freq

    def Feature_testID_High_Freq(self, df) :
        '''
        전체 유저에 대해 `testID`가 평균 이상으로 노출되었는지 여부
        '''
        df['testID_cnt'] = self.Feature_testID_Count(df)
        testID_High_Freq = np.where(df['testID_cnt'] >= df['testID'].value_counts().mean(), 1, 0).astype('int8')
        return testID_High_Freq

    def Feature_testCode_High_Freq(self, df) :
        '''
        전체 유저에 대해 `testCode`가 평균 이상으로 노출되었는지 여부
        '''
        df['testCode_cnt'] = self.Feature_testCode_Count(df)
        testCode_High_Freq = np.where(df['testCode_cnt'] >= df['testCode'].value_counts().mean(), 1, 0).astype('int8')
        return testCode_High_Freq

    def Feature_testNum_High_Freq(self, df) :
        '''
        전체 유저에 대해 `testNum`가 평균 이상으로 노출되었는지 여부
        '''
        df['testNum_cnt'] = self.Feature_testNum_Count(df)
        testNum_High_Freq = np.where(df['testNum_cnt'] >= df['testNum'].value_counts().mean(), 1, 0).astype('int8')
        return testNum_High_Freq

    def Feature_problemID_High_Freq(self, df) :
        '''
        전체 유저에 대해 `problemID`가 평균 이상으로 노출되었는지 여부
        '''
        df['problemID_cnt'] = self.Feature_problemID_Count(df)
        problemID_High_Freq = np.where(df['problemID_cnt'] >= df['problemID'].value_counts().mean(), 1, 0).astype('int8')
        return problemID_High_Freq

    def Feature_Tag_High_Freq(self, df) :
        '''
        전체 유저에 대해 `KnowledgeTag`가 평균 이상으로 노출되었는지 여부
        '''
        df['Tag_cnt'] = self.Feature_Tag_Count(df)
        Tag_High_Freq = np.where(df['Tag_cnt'] >= df['KnowledgeTag'].value_counts().mean(), 1, 0).astype('int8')
        return Tag_High_Freq
    
    ### 과거 정보 활용한 변수
    def Feature_User_Past_Solved(self, df) :
        '''
        유저가 해당 문제를 이전에 풀었던 횟수
        '''
        df['user_past'] = df.groupby(['userID', 'assessmentItemID'])['answerCode'].shift(1).fillna(0).astype('int8')
        User_Past_Solved = df.groupby(['userID', 'assessmentItemID'])['user_past'].cumsum().astype('int8')
        return User_Past_Solved.values
    
    def Feature_Relative_Correct_Rate(self, df) :
        '''
        상대적인 정답률 (`answerCode` - `itemID_acc`)
        '''
        Item_Acc = self.Feature_Item_Acc(df)
        Relative_Correct_Rate = df['answerCode'].values - Item_Acc
        return Relative_Correct_Rate
    
    def Feature_Is_Correct_Before1(self, df) :
        '''
        1번째 이전 문제의 정답 여부
        '''
        df['testID'] = self.Feature_testID(df)
        Is_Correct_Before1 = df.groupby(['userID', 'testID'], as_index = False)['answerCode'].shift(1).fillna(1).astype('int8')
        return Is_Correct_Before1.values

    def Feature_Correct_Rate_Before1(self, df) :
        '''
        1번째 이전 문제의 정답률
        '''
        df['testID'] = self.Feature_testID(df)
        df['itemID_acc'] = self.Feature_Item_Acc(df)
        Correct_Rate_Before1 = df.groupby(['userID', 'testID'], as_index = False)['itemID_acc'].shift(1).fillna(1)
        return Correct_Rate_Before1.values

    def Feature_Relative_Correct_Rate_Before1(self, df) :
        '''
        1번째 이전 문제의 상대적인 정답률
        '''
        df['testID'] = self.Feature_testID(df)
        df['relative_correct_rate'] = self.Feature_Relative_Correct_Rate(df)
        Relative_Correct_Rate_Before1 = df.groupby(['userID', 'testID'], as_index = False)['relative_correct_rate'].shift(1).fillna(1)
        return Relative_Correct_Rate_Before1.values

    def Feature_Is_Correct_Before2(self, df) :
        '''
        2번째 이전 문제의 정답 여부
        '''
        df['testID'] = self.Feature_testID(df)
        Is_Correct_Before2 = df.groupby(['userID', 'testID'], as_index = False)['answerCode'].shift(2).fillna(1).astype('int8')
        return Is_Correct_Before2.values

    def Feature_Correct_Rate_Before2(self, df) :
        '''
        2번째 이전 문제의 정답률
        '''
        df['testID'] = self.Feature_testID(df)
        df['itemID_acc'] = self.Feature_Item_Acc(df)
        Correct_Rate_Before2 = df.groupby(['userID', 'testID'], as_index = False)['itemID_acc'].shift(2).fillna(1)
        return Correct_Rate_Before2.values

    def Feature_Relative_Correct_Rate_Before2(self, df) :
        '''
        2번째 이전 문제의 상대적인 정답률
        '''
        df['testID'] = self.Feature_testID(df)
        df['relative_correct_rate'] = self.Feature_Relative_Correct_Rate(df)
        Relative_Correct_Rate_Before2 = df.groupby(['userID', 'testID'], as_index = False)['relative_correct_rate'].shift(2).fillna(1)
        return Relative_Correct_Rate_Before2.values

    def Feature_Is_Correct_Before3(self, df) :
        '''
        3번째 이전 문제의 정답 여부
        '''
        df['testID'] = self.Feature_testID(df)
        Is_Correct_Before3 = df.groupby(['userID', 'testID'], as_index = False)['answerCode'].shift(3).fillna(1).astype('int8')
        return Is_Correct_Before3.values

    def Feature_Correct_Rate_Before3(self, df) :
        '''
        3번째 이전 문제의 정답률
        '''
        df['testID'] = self.Feature_testID(df)
        df['itemID_acc'] = self.Feature_Item_Acc(df)
        Correct_Rate_Before3 = df.groupby(['userID', 'testID'], as_index = False)['itemID_acc'].shift(3).fillna(1)
        return Correct_Rate_Before3.values

    def Feature_Relative_Correct_Rate_Before3(self, df) :
        '''
        3번째 이전 문제의 상대적인 정답률
        '''
        df['testID'] = self.Feature_testID(df)
        df['relative_correct_rate'] = self.Feature_Relative_Correct_Rate(df)
        Relative_Correct_Rate_Before3 = df.groupby(['userID', 'testID'], as_index = False)['relative_correct_rate'].shift(3).fillna(1)
        return Relative_Correct_Rate_Before3.values

    def Feature_Is_Correct_Before4(self, df) :
        '''
        4번째 이전 문제의 정답 여부
        '''
        df['testID'] = self.Feature_testID(df)
        Is_Correct_Before4 = df.groupby(['userID', 'testID'], as_index = False)['answerCode'].shift(4).fillna(1).astype('int8')
        return Is_Correct_Before4.values

    def Feature_Correct_Rate_Before4(self, df) :
        '''
        4번째 이전 문제의 정답률
        '''
        df['testID'] = self.Feature_testID(df)
        df['itemID_acc'] = self.Feature_Item_Acc(df)
        Correct_Rate_Before4 = df.groupby(['userID', 'testID'], as_index = False)['itemID_acc'].shift(4).fillna(1)
        return Correct_Rate_Before4.values

    def Feature_Relative_Correct_Rate_Before4(self, df) :
        '''
        4번째 이전 문제의 상대적인 정답률
        '''
        df['testID'] = self.Feature_testID(df)
        df['relative_correct_rate'] = self.Feature_Relative_Correct_Rate(df)
        Relative_Correct_Rate_Before4 = df.groupby(['userID', 'testID'], as_index = False)['relative_correct_rate'].shift(4).fillna(1)
        return Relative_Correct_Rate_Before4.values

    def Feature_Is_Correct_Before5(self, df) :
        '''
        5번째 이전 문제의 정답 여부
        '''
        df['testID'] = self.Feature_testID(df)
        Is_Correct_Before5 = df.groupby(['userID', 'testID'], as_index = False)['answerCode'].shift(5).fillna(1).astype('int8')
        return Is_Correct_Before5.values

    def Feature_Correct_Rate_Before5(self, df) :
        '''
        5번째 이전 문제의 정답률
        '''
        df['testID'] = self.Feature_testID(df)
        df['itemID_acc'] = self.Feature_Item_Acc(df)
        Correct_Rate_Before5 = df.groupby(['userID', 'testID'], as_index = False)['itemID_acc'].shift(5).fillna(1)
        return Correct_Rate_Before5.values

    def Feature_Relative_Correct_Rate_Before5(self, df) :
        '''
        5번째 이전 문제의 상대적인 정답률
        '''
        df['testID'] = self.Feature_testID(df)
        df['relative_correct_rate'] = self.Feature_Relative_Correct_Rate(df)
        Relative_Correct_Rate_Before5 = df.groupby(['userID', 'testID'], as_index = False)['relative_correct_rate'].shift(5).fillna(1)
        return Relative_Correct_Rate_Before5.values
    
    ### ELO Rating 변수
    def Feature_ELO_Theta(self, df) :
        '''
        - ELO Rating의 유저의 theta 추정 값 (theta_estimate_json 에 저장)
        '''
        theta_estimate, _ = ELO(df)
        theta = df['userID'].astype(str).map(theta_estimate)
        return theta
    
    def Feature_ELO_Beta(self, df) :
        '''
        - ELO Rating의 문제의 beta 추정 값 (beta_estimate_json 에 저장)
        '''
        _, beta_estimate = ELO(df)
        beta = df['assessmentItemID'].map(beta_estimate)
        return beta
