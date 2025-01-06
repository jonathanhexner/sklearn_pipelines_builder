from catboost import CatBoostClassifier, Pool, CatBoostRegressor

class CatBoostModelFactory:
    @staticmethod
    def create(model_type: str, *args, **kwargs):
        if model_type == 'catboost_regressor':
            return CatBoostRegressor(*args, **kwargs)
        elif model_type == 'catboost_classifier':
            return CatBoostClassifier(*args, **kwargs)
        raise ValueError(f"Unsupported model_type {model_type}")
