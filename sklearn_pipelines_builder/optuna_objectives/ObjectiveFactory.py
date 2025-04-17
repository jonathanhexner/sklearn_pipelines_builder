
class ObjectiveFactory:
    @staticmethod
    def create_objective(config):
        model_config = config.get('model_config')
        model_name = model_config.get("element_type")
        # TODO: add this
        # if model_name == "xgboost":
        #     return XGBoostObjective(config)
        # elif model_name == "lightgbm":
        #     return LightGBMObjective(config)
        if model_name.startswith("catboost"):
            from sklearn_pipelines_builder.optuna_objectives.CatBoostObjective import CatBoostObjective
            return CatBoostObjective(config)
        elif model_name == "nn":
            from sklearn_pipelines_builder.optuna_objectives.NNObjective import NNObjective
            return NNObjective(config)
        else:
            raise ValueError(f"Unsupported model_name: {model_name}")
