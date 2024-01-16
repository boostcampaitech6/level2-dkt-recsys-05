import json
import os
from optuna import Trial
from xgboost import XGBClassifier
from xgboost import plot_importance
from optuna.integration.xgboost import XGBoostPruningCallback
from optuna.samplers import TPESampler
from sklearn.metrics import roc_auc_score, accuracy_score, log_loss
import optuna
import numpy as np
import matplotlib.pyplot as plt

from config import XGBoostConfig


class XGBoost:
    def __init__(
        self,
        config: XGBoostConfig,
        use_columns: list[str],
        X_train,
        y_train,
        X_valid,
        y_valid,
        test_GB,
        exp_code: str,
    ):
        self.config = config
        self.use_columns = use_columns

        self.X_train = X_train
        self.y_train = y_train
        self.X_valid = X_valid
        self.y_valid = y_valid
        self.test_GB = test_GB

        self.exp_code = exp_code

        self.best_params = {
            "booster": "dart",
            "max_depth": 13,
            "learning_rate": 0.05,
            "min_child_weight": 6,
            "gamma": 1,
            "colsample_bytree": 0.5,
            "lambda": 10,
            "alpha": 1,
            "subsample": 1.0,
            "max_delta_step": 5,
        }

        self.init_experiment()

    def _write_path(self, dir, path):
        if not os.path.exists(dir):
            os.makedirs(dir)

        return os.path.join(dir, path)

    def init_experiment(self):
        base = "exp/"
        dir_list = os.listdir(base)
        last_exp_seq = 0
        for dir in dir_list:
            if dir.split("_")[0].isdigit():
                last_exp_seq = max(last_exp_seq, int(dir.split("_")[0]))

        self.exp_code = f"{last_exp_seq + 1}_{self.exp_code}"

        write_path = self._write_path(
            f"exp/{self.exp_code}/",
            f"exp_{self.exp_code}.json",
        )

        exp = {
            "exp_code": self.exp_code,
            "config": self.config.dict(),
            "use_columns": self.use_columns,
        }

        with open(write_path, "w", encoding="utf8") as w:
            json.dump(exp, w, ensure_ascii=False, indent=2)

    def update_experiment(self, key, value):
        write_path = self._write_path(
            f"exp/{self.exp_code}/",
            f"exp_{self.exp_code}.json",
        )

        with open(write_path, "r", encoding="utf8") as r:
            exp = json.load(r)

        exp[key] = value

        with open(write_path, "w", encoding="utf8") as w:
            json.dump(exp, w, ensure_ascii=False, indent=2)

    def hpo_optimizer(self, trial: Trial):
        param = {
            "booster": trial.suggest_categorical("booster", ["gbtree", "dart"]),
            "max_depth": trial.suggest_int("max_depth", 1, 15),
            "learning_rate": trial.suggest_categorical(
                "learning_rate", [1e-3, 0.01, 0.05, 0.1, 0.5]
            ),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "gamma": trial.suggest_categorical("gamma", [1e-5, 1e-3, 1, 5, 10]),
            "colsample_bytree": trial.suggest_categorical(
                "colsample_bytree", [0.1, 0.5, 1]
            ),
            "lambda": trial.suggest_categorical("lambda", [1e-5, 1e-3, 1, 5, 10]),
            "alpha": trial.suggest_categorical("alpha", [1e-5, 1e-3, 1, 5, 10]),
            "subsample": trial.suggest_categorical("subsample", [0.6, 0.7, 0.8, 1.0]),
            "max_delta_step": trial.suggest_categorical(
                "max_delta_step", [0.1, 0.5, 1, 5, 10]
            ),
        }

        pruning_callback = XGBoostPruningCallback(trial, "validation_1-" + "auc")

        xgb_model = XGBClassifier(
            **param,
            **self.config.dict(),
        ).fit(
            self.X_train,
            self.y_train,
            eval_set=[(self.X_train, self.y_train), (self.X_valid, self.y_valid)],
            verbose=300,
            callbacks=[pruning_callback],
        )

        return roc_auc_score(self.y_valid, xgb_model.predict_proba(self.X_valid)[:, 1])

    def hpo_start(self):
        study = optuna.create_study(
            direction="maximize", sampler=TPESampler(seed=self.config.random_state)
        )
        study.optimize(
            lambda trial: self.hpo_optimizer(trial),
            show_progress_bar=True,
            n_trials=100,
        )
        self.best_params = study.best_trial.params
        print(
            f"Best trial : score {study.best_trial.value}, \n params = {study.best_trial.params} \n"
        )

    def train_start(self):
        self.update_experiment("best_params", self.best_params)

        model = XGBClassifier(**self.best_params, **self.config.dict()).fit(
            self.X_train,
            self.y_train,
            eval_set=[(self.X_train, self.y_train), (self.X_valid, self.y_valid)],
            verbose=100,
        )

        proba = model.predict_proba(self.X_valid)[:, 1]
        roc_auc = roc_auc_score(self.y_valid, proba)
        accuracy = accuracy_score(self.y_valid, np.where(proba >= 0.5, 1, 0))
        logloss = log_loss(self.y_valid, proba)

        print(
            f"ROC-AUC Score : {roc_auc:.4f} / Accuracy : {accuracy:.4f} / Logloss : {logloss:.4f}"
        )

        self.update_experiment("roc_auc", roc_auc)
        self.update_experiment("accuracy", accuracy)
        self.update_experiment("logloss", logloss)

        proba = model.predict_proba(self.test_GB)[:, 1]

        self.save_feature_importance_plot(model)
        self.save_model(model)
        self.save_output(proba)

    def save_feature_importance_plot(self, model):
        _, axes = plt.subplots(nrows=1, ncols=2, figsize=(20, 10))
        plot_importance(model, ax=axes[0], importance_type="gain")
        axes[0].set_title("Feature Importance (type = gain)")
        plot_importance(model, ax=axes[1], importance_type="weight")
        axes[1].set_title("Feature Importance (type = weight)")

        write_path = self._write_path(
            f"exp/{self.exp_code}/",
            f"feature_importances_{self.exp_code}.png",
        )

        plt.tight_layout()
        plt.savefig(write_path)

    def save_model(self, model):
        write_path = self._write_path(
            f"exp/{self.exp_code}/",
            f"model_{self.exp_code}.model",
        )

        model.save_model(write_path)

    def save_output(self, proba):
        write_path = self._write_path(
            f"exp/{self.exp_code}/",
            f"output_{self.exp_code}.csv",
        )

        with open(write_path, "w", encoding="utf8") as w:
            w.write("id,prediction\n")
            for id, p in enumerate(proba):
                w.write("{},{}\n".format(id, p))
