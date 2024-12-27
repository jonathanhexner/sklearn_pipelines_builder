def objective_cv(trial, config={}):
    model_params = {
        "iterations": trial.suggest_int("iterations", 500, 2000),
        "depth": trial.suggest_int("depth", 4, 10),
        "learning_rate": trial.suggest_float("learning_rate", 1e-3, 1.0, log=True),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1, 10),
        'border_count': trial.suggest_int("border_count", 4, 500),
        'bagging_temperature': trial.suggest_float("bagging_temperature", 0, 100),
        'scale_pos_weight': trial.suggest_float("scale_pos_weight", 1, 30),

    }
    weight_factor = trial.suggest_float("weight_factor", 0.5, 10.0, log=True)
    # Compute initial sample weights based on class imbalance
    base_weights = compute_sample_weight("balanced", y)
    adjusted_weights = base_weights * weight_factor  # Adjust weights by the factor


    # string_column= options.get('string_column')
    # string_features = options.get('string_features')
    # X= options.get('X')
    #
    #
    # X[string_column] = X[string_column].astype(str)
    # string_features = string_features+[string_column]



    model = CatBoostClassifier(**model_params, cat_features =string_features, verbose=0)
    score = cross_val_score(model, X, y, cv=3, scoring="roc_auc",
                            fit_params={'sample_weight': adjusted_weights}).mean()
    return score
