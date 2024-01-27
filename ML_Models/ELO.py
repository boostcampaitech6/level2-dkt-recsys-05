import json
import numpy as np 
import pandas as pd
from tqdm import tqdm

def ELO(data, user_feature_name = 'userID', granularity_feature_name = 'assessmentItemID', compute_estimations = False, nb_rows_training = None) :
    '''
    Competition : Kaggle Riiid Answer Correctness Prediction
    ELO Rating Reference : https://www.kaggle.com/code/stevemju/riiid-simple-elo-rating/notebook

    Description
    - `theta` represents the global level of the student
    - `beta` represents the difficulty of the item (items are represented by `content_id` in that case but you could change it by modifying the `granularity_feature_name` parameters)
    - `left_asymptote` represents the probability of correctly answering a question by chance. The `correct_answer` from the questions dataset has 4 possible values, so this probability is at least 1/4

    Remarks
    - I used a different learning rate for thetas and for betas. The learning rate for thetas has a floor value to keep it dynamic even after many answers as the student level is always evolving contrary to the difficulty of questions.
    - The learning rate for theta update is absolutely arbitrary (I picked one that looked good, nothing more). The learning rate for betas is one that was recommended this paper but I'm sure there's a better one.
    - I tried to compute a theta by part but performance is slightly worse (0.764).
    - I tried both a theta by part and a global theta, which I combined with proportions based on the nb_answers of the part, and performance is slightly better (0.767).

    Possible improvements:
    - Fine tune learning rates (simply changing the LR for betas made the score go from 0.749 to 0.766)
    - Learning rate function of time: if the student hasn't played for a long time, we might want to increase the LR to reestimate his/her level quickly
    - Add a coefficient on `theta` computation based on the time to answer (a good and quick answer is probably more indicative of mastery than a good and slow answer)
    Investigate on extreme `theta` and `beta` values
    - Using betas & thetas as model features could be interesting, as well as estimating betas by tags
    --------------------------------------------------------------------------------------------------
    Parameter
    data                : Row Data의 파일 경로
    compute_estimations : theta와 beta에 대한 추정 여부 (처음 한 번 True로 실행)
    nb_rows_training    : 학습에 사용할 행의 수 (default = None)
    '''
    ### ELO functions
    def get_new_theta(is_good_answer, beta, left_asymptote, theta, nb_previous_answers) :
        return theta + learning_rate_theta(nb_previous_answers) * (
            is_good_answer - probability_of_good_answer(theta, beta, left_asymptote))

    def get_new_beta(is_good_answer, beta, left_asymptote, theta, nb_previous_answers) :
        return beta - learning_rate_beta(nb_previous_answers) * (
            is_good_answer - probability_of_good_answer(theta, beta, left_asymptote))

    def learning_rate_theta(nb_answers) :
        return max(0.3 / (1 + 0.01 * nb_answers), 0.04)

    def learning_rate_beta(nb_answers) :
        return 1 / (1 + 0.05 * nb_answers)

    def probability_of_good_answer(theta, beta, left_asymptote) :
        return left_asymptote + (1 - left_asymptote) * sigmoid(theta - beta)

    def sigmoid(x) :
        return 1 / (1 + np.exp(-x))
    
    ### Parameters Estimation
    def estimate_parameters(answers_df, user_feature_name, granularity_feature_name) :
        item_parameters = {
            granularity_feature_value : {'beta' : 0, 'item_nb_answers' : 0}
            for granularity_feature_value in np.unique(answers_df[granularity_feature_name])}
        user_parameters = {
            user_id : {'theta' : 0, 'user_nb_answers' : 0}
            for user_id in np.unique(answers_df[user_feature_name])}

        print('Parameter estimation is starting...')

        for user_id, item_id, left_asymptote, answerCode in tqdm(
            zip(answers_df[user_feature_name].values, answers_df[granularity_feature_name].values,
                answers_df['left_asymptote'].values, answers_df['answerCode'].values)
            ) :
                theta = user_parameters[user_id]['theta']
                beta = item_parameters[item_id]['beta']

                item_parameters[item_id]['beta'] = get_new_beta(
                    answerCode, beta, left_asymptote, theta, item_parameters[item_id]['item_nb_answers'],)
                user_parameters[user_id]['theta'] = get_new_theta(
                    answerCode, beta, left_asymptote, theta, user_parameters[user_id]['user_nb_answers'],)
        
                item_parameters[item_id]['item_nb_answers'] += 1
                user_parameters[user_id]['user_nb_answers'] += 1

        print(f'Theta & beta estimations on {granularity_feature_name} are completed.')
        return user_parameters, item_parameters
    
    ### Update Parameters
    def update_parameters(answers_df, user_parameters, item_parameters, user_feature_name, granularity_feature_name) :
        for user_id, item_id, left_asymptote, answerCode in tqdm(zip(
            answers_df[user_feature_name].values, 
            answers_df[granularity_feature_name].values, 
            answers_df['left_asymptote'].values, 
            answers_df['answerCode'].values)
        ) :
            if user_id not in user_parameters :
                user_parameters[user_id] = {'theta' : 0, 'user_nb_answers' : 0}
            if item_id not in item_parameters :
                item_parameters[item_id] = {'beta' : 0, 'item_nb_answers' : 0}

            theta = user_parameters[user_id]['theta']
            beta = item_parameters[item_id]['beta']

            user_parameters[user_id]['theta'] = get_new_theta(
                answerCode, beta, left_asymptote, theta, user_parameters[user_id]['user_nb_answers'])
            item_parameters[item_id]['beta'] = get_new_beta(
                answerCode, beta, left_asymptote, theta, item_parameters[item_id]['item_nb_answers'])

            user_parameters[user_id]['user_nb_answers'] += 1
            item_parameters[item_id]['item_nb_answers'] += 1
        
        return user_parameters, item_parameters
    
    ### Probability Estimation
    def estimate_probas(test_df, user_parameters, item_parameters, user_feature_name, granularity_feature_name) :
        probability_of_success_list = []
    
        for user_id, item_id, left_asymptote in tqdm(
            zip(test_df[user_feature_name].values, test_df[granularity_feature_name].values, test_df['left_asymptote'].values)
        ) :
            theta = user_parameters[user_id]['theta'] if user_id in user_parameters else 0
            beta = item_parameters[item_id]['beta'] if item_id in item_parameters else 0

            probability_of_success_list.append(probability_of_good_answer(theta, beta, left_asymptote))

        return probability_of_success_list
    
    if compute_estimations :
        if type(data) == 'str' : # 데이터 경로로 주어지면
            train_data = pd.read_csv(
                filepath_or_buffer = data, 
                usecols = ['assessmentItemID', 'userID', 'answerCode'], 
                dtype = {'answerCode' : 'int8'},
                nrows = nb_rows_training)
        elif type(data) == pd.core.frame.DataFrame : # 데이터프레임으로 주어지면
            train_data = data.copy()
        else : 
            raise ValueError
        
        training = train_data[train_data['answerCode'] != -1]
        training['left_asymptote'] = 0
    
        print(f'Dataset of shape {training.shape}')
        print(f'Columns are {list(training.columns)}')

        user_parameters, item_parameters = estimate_parameters(training, user_feature_name, granularity_feature_name)
        theta_estimate = {int(user) : theta['theta'] for user, theta in user_parameters.items()}
        beta_estimate = {str(item) : beta['beta'] for item, beta in item_parameters.items()}
        
        with open('./data/theta_estimate.json', 'w') as f : 
            json.dump(theta_estimate, f, indent = 4)
        with open('./data/beta_estimate.json', 'w') as f : 
            json.dump(beta_estimate, f, indent = 4)
 
        print('Successfully Write User (theta) / Item (beta) parameter file.')
        return theta_estimate, beta_estimate
        
    
    else :
        with open('/data/ephemeral/data/theta_estimate.json') as f :
            theta_estimate = json.load(f)
        with open('/data/ephemeral/data/beta_estimate.json') as f :
            beta_estimate = json.load(f)
        print('Successfully Read User (theta) / Item (beta) parameter file.')
        return theta_estimate, beta_estimate