# from sklearn.model_selection import cross_val_score
import numpy as np
from sklearn.model_selection import StratifiedKFold # , cross_val_score
from sklearn_pipelines_builder.imputers.SelectBestImputer import global_config
from sklearn_pipelines_builder.validation.cross_val_score import cross_val_score
from sklearn_pipelines_builder.models.CatBoostWrapper import CatBoostWrapper
from sklearn_pipelines_builder.infrastructure.Config import Config
global_config = Config()

def objective_catboost(trial, X, y):
    model_params = {
        "iterations": trial.suggest_int("iterations", 500, 2000),
        "depth": trial.suggest_int("depth", 4, 10),
        "learning_rate": trial.suggest_float("learning_rate", 1e-3, 1.0, log=True),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1, 10),
        'border_count': trial.suggest_int("border_count", 4, 500),
        'bagging_temperature': trial.suggest_float("bagging_temperature", 0, 100),
        'scale_pos_weight': trial.suggest_float("scale_pos_weight", 1, 30),
        'verbose': 0
    }

    # pipeline = create_pipeline(pipeline_config)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # Set scoring as AUC

    # scoring = "roc_auc"
    model = CatBoostWrapper(model_params)
    # score = np.mean(my_cross_val_score(X, y, pipeline, n_splits=3))
    cv_scores = cross_val_score( X, y, model, cv=cv, scoring=global_config.scoring)
    trial.set_user_attr("trained_model", model)

    return np.mean(cv_scores)
