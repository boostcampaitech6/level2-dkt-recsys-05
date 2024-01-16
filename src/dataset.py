from sklearn.calibration import LabelEncoder

import pandas as pd
import numpy as np

from config import Config


def get_data(config: Config):
    train = pd.read_csv(
        config.data_dir + "train_data.csv",
        dtype={
            "userID": "int16",
            "answerCode": "int8",
            "KnowledgeTag": "int16",
        },
        parse_dates=["Timestamp"],
    )
    train = train.sort_values(by=["userID", "Timestamp"]).reset_index(drop=True)

    data = pd.read_parquet(config.data_dir + config.data_preprocessed_parquet)

    valid_indices = (
        train.reset_index()
        .groupby("userID", as_index=False)
        .last()
        .set_index("index")
        .index
    )

    train_le, test_GB = data[data["answerCode"] != -1], data[
        data["answerCode"] == -1
    ].drop(columns="answerCode")

    valid_indices = (
        train.reset_index()
        .groupby("userID", as_index=False)
        .last()
        .set_index("index")
        .index
    )

    obj_col = ["assessmentItemID", "testID"]
    for col in obj_col:
        le = LabelEncoder()
        train_le[col] = le.fit_transform(train_le[col])
        for label in test_GB[col].unique():
            if label not in le.classes_:
                le.classes_ = np.append(le.classes_, label)
        test_GB[col] = le.transform(test_GB[col])

    train_GB = train_le.loc[~train_le.index.isin(valid_indices)]
    valid_GB = train_le.loc[train_le.index.isin(valid_indices)]

    X_train, y_train = train_GB[config.use_columns], train_GB["answerCode"]
    X_valid, y_valid = valid_GB[config.use_columns], valid_GB["answerCode"]
    test_GB = test_GB[config.use_columns]

    return (
        X_train,
        y_train,
        X_valid,
        y_valid,
    )
