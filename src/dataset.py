from sklearn.calibration import LabelEncoder

import pandas as pd
import numpy as np

from config import Config


def get_data(config: Config):
    data = pd.read_parquet(config.data_dir + config.data_preprocessed_parquet)

    train_le, test_le = data[data["answerCode"] != -1], data[
        data["answerCode"] == -1
    ].drop(columns="answerCode")
    valid_indices = set(data[data["answerCode"] != -1].index).intersection(
        set(
            data.reset_index()
            .groupby("userID", as_index=False)
            .last()
            .set_index("index")
            .index
        )
    )

    obj_col = ["assessmentItemID", "testID"]
    for col in obj_col:
        le = LabelEncoder()
        train_le[col] = le.fit_transform(train_le[col])
        for label in test_le[col].unique():
            if label not in le.classes_:
                le.classes_ = np.append(le.classes_, label)
        test_le[col] = le.transform(test_le[col])

    train_GB = train_le.loc[~train_le.index.isin(valid_indices)]
    valid_GB = train_le.loc[train_le.index.isin(valid_indices)]

    X_train, y_train = train_GB[config.use_columns], train_GB["answerCode"]
    X_valid, y_valid = valid_GB[config.use_columns], valid_GB["answerCode"]
    test_GB = test_le[config.use_columns]

    print(f"X_train.shape: {X_train.shape}")
    print(f"y_train.shape: {y_train.shape}")
    print(f"X_valid.shape: {X_valid.shape}")
    print(f"y_valid.shape: {y_valid.shape}")
    print(f"test_GB.shape: {test_GB.shape}")

    return (
        X_train,
        y_train,
        X_valid,
        y_valid,
        test_GB,
    )
