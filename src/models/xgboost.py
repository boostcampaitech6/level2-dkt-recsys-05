from optuna import Trial
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier
from xgboost import plot_importance
from optuna.integration.xgboost import XGBoostPruningCallback
from optuna.samplers import TPESampler
from sklearn.metrics import roc_auc_score, accuracy_score, log_loss
import optuna
import numpy as np
import matplotlib.pyplot as plt
from wandb.xgboost import WandbCallback

from models._main import BoostingBasedModel
from config import Config
from utils.common import new_experiment, get_path, update_experiment


class XGBoost(BoostingBasedModel):
    def __init__(
        self,
        config: Config,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_valid: pd.DataFrame,
        y_valid: pd.Series,
        test_GB: pd.DataFrame,
        exp_code: str,
    ):
        self.config = config.xgb
        self.use_columns = config.use_columns

        self.X_train = X_train
        self.y_train = y_train
        self.X_valid = X_valid
        self.y_valid = y_valid
        self.test_GB = test_GB

        self.exp_code = exp_code

        self.best_params = config.best_params
        self.hpo_config = config.hpo
        self.fold_config = config.fold

        new_experiment(
            exp_code,
            {
                "exp_code": self.exp_code,
                "config": self.config.dict(),
                "use_columns": self.use_columns,
            },
        )

    def _hpo_optimizer(self, trial: Trial):
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

    def hpo(self):
        study = optuna.create_study(
            direction="maximize", sampler=TPESampler(seed=self.config.random_state)
        )

        study.optimize(
            lambda trial: self._hpo_optimizer(trial),
            show_progress_bar=True,
            n_trials=self.hpo_config.n_trials,
        )

        self.best_params = study.best_trial.params

        print(
            f"Best trial : score {study.best_trial.value}, \n params = {study.best_trial.params} \n"
        )

    def train(self):
        update_experiment(self.exp_code, {"best_params": self.best_params})

        if self.fold_config.skip:
            model, proba, _, _, _ = self._train(
                self.X_train,
                self.y_train,
                self.X_valid,
                self.y_valid,
                self.test_GB,
            )

            self.save_feature_importance_plot(model)
            self.save_model(model)
            self.save_output(proba)
        else:
            skf = StratifiedKFold(
                n_splits=self.fold_config.n_splits,
                shuffle=True,
                random_state=self.config.random_state,
            )

            valid_auc = []
            proba_df = pd.DataFrame()

            for fold, (train_idx, valid_idx) in enumerate(
                skf.split(self.X_train, self.y_train)
            ):
                print(f"Fold {fold} Start!")
                X_train, X_valid = (
                    self.X_train.iloc[train_idx],
                    self.X_train.iloc[valid_idx],
                )
                y_train, y_valid = (
                    self.y_train.iloc[train_idx],
                    self.y_train.iloc[valid_idx],
                )

                model, proba, roc_auc, accuracy, logloss = self._train(
                    X_train,
                    y_train,
                    X_valid,
                    y_valid,
                    self.test_GB,
                )

                valid_auc.append(roc_auc)
                proba_df[f"fold_{fold}"] = proba

                self.save_feature_importance_plot(model, fold)
                self.save_model(model, fold)

                print(f"Fold {fold} Done!")
                print(
                    f"ROC-AUC Score : {roc_auc:.4f} / Accuracy : {accuracy:.4f} / Logloss : {logloss:.4f}"
                )

            proba_df["mean"] = proba_df.mean(axis=1)
            proba_df["std"] = proba_df.std(axis=1)

            update_experiment(
                self.exp_code,
                {
                    "valid_auc": valid_auc,
                    "mean_auc": np.mean(valid_auc),
                    "std_auc": np.std(valid_auc),
                },
            )

            self.save_output(proba_df["mean"])

    def _train(self, X_train, y_train, X_valid, y_valid, test_GB):
        pruning_callback = WandbCallback(log_model=True)

        model = XGBClassifier(**self.best_params, **self.config.dict()).fit(
            X_train,
            y_train,
            eval_set=[(X_train, y_train), (X_valid, y_valid)],
            verbose=100,
            callbacks=[pruning_callback],
        )

        proba = model.predict_proba(X_valid)[:, 1]
        roc_auc = roc_auc_score(y_valid, proba)
        accuracy = accuracy_score(y_valid, np.where(proba >= 0.5, 1, 0))
        logloss = log_loss(y_valid, proba)

        print(
            f"ROC-AUC Score : {roc_auc:.4f} / Accuracy : {accuracy:.4f} / Logloss : {logloss:.4f}"
        )

        update_experiment(
            self.exp_code,
            {
                "roc_auc": roc_auc,
                "accuracy": accuracy,
                "logloss": logloss,
            },
        )

        proba = model.predict_proba(test_GB)[:, 1]

        return model, proba, roc_auc, accuracy, logloss

    def save_feature_importance_plot(self, model, seq=0):
        _, axes = plt.subplots(nrows=1, ncols=2, figsize=(20, 10))
        plot_importance(model, ax=axes[0], importance_type="gain")
        axes[0].set_title("Feature Importance (type = gain)")
        plot_importance(model, ax=axes[1], importance_type="weight")
        axes[1].set_title("Feature Importance (type = weight)")

        plt.tight_layout()
        plt.savefig(
            get_path(
                f"exp/{self.exp_code}/",
                f"feature_importances_{self.exp_code}_F{seq}.png",
            )
        )

    def save_model(self, model, seq=0):
        model.save_model(
            get_path(
                f"exp/{self.exp_code}/",
                f"model_{self.exp_code}_F{seq}.model",
            )
        )

    def save_output(self, proba):
        with open(
            get_path(
                f"exp/{self.exp_code}/",
                f"output_{self.exp_code}.csv",
            ),
            "w",
            encoding="utf8",
        ) as w:
            w.write("id,prediction\n")
            for id, p in enumerate(proba):
                w.write("{},{}\n".format(id, p))
