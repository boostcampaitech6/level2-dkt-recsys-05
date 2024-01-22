from abc import *

from config import ModelType


class BoostingBasedModel(metaclass=ABCMeta):
    def __init__(self, name, model):
        self.name = name
        self.model = model

    @abstractmethod
    def train(self, *args, **kwargs):
        pass
    
    @abstractmethod
    def hpo(self, *args, **kwargs):
        pass


def get_model(model_type: ModelType, *args, **kwargs) -> BoostingBasedModel:
    if model_type == ModelType.XGBoost:
        from models.xgboost import XGBoost
        return XGBoost(*args, **kwargs)
    elif model_type == ModelType.LightGBM:
        from models.lightgbm import LightGBM
        return LightGBM(*args, **kwargs)
    else:
        raise NotImplementedError(f'{model_type} is not implemented')
