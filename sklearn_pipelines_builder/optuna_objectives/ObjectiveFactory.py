
class ObjectiveFactory:
    @staticmethod
    def create_objective(config):
        model_name = config.get("model_name")
        if model_name == "xgboost":
            return XGBoostObjective(config)
        elif model_name == "lightgbm":
            return LightGBMObjective(config)
        elif model_name == "catboost":
            from sklearn_pipelines_builder.optuna_objectives.CatBoostObjective import CatBoostObjective
            return CatBoostObjective(config)
        else:
            raise ValueError(f"Unsupported model_name: {model_name}")
