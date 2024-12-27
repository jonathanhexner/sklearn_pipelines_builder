def my_cross_val_score(X, y, pipeline, n_splits=3):
    # Manual cross-validation
    kf = StratifiedKFold(n_splits=3)
    scores = []

    for train_index, val_index in kf.split(X, y):
        X_train, X_val = X.iloc[train_index], X.iloc[val_index]
        y_train, y_val = y.iloc[train_index], y.iloc[val_index]

        # Fit the pipeline on the training fold
        pipeline.fit(X_train, y_train)

        # Get predictions on the validation fold
        y_val_predict = pipeline.predict_proba(X_val)

        # Calculate AUC score
        fold_auc = roc_auc_score(y_val, y_val_predict[:, 1])
        scores.append(fold_auc)
    return scores

# Calculate average score
