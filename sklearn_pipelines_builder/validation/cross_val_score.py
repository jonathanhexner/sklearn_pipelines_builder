from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.metrics import get_scorer
from sklearn_pipelines_builder.infrastructure.Config import Config
import numpy as np

def cross_val_score(pipeline, X, y, cv, scoring):
    # Manual cross-validation
    scores = []

    for train_index, val_index in cv.split(X, y):
        print(train_index, val_index)
        X_train, X_val = X.iloc[train_index], X.iloc[val_index]
        y_train, y_val = y.iloc[train_index], y.iloc[val_index]

        # Fit the pipeline on the training fold
        pipeline.fit(X_train, y_train)

        scorer = get_scorer(Config().scoring)
        scores.append(scorer(pipeline, X_val, y_val))

        # # Get predictions on the validation fold
        # y_val_predict = pipeline.predict_proba(X_val)
        #
        # # Calculate AUC score
        # fold_auc = roc_auc_score(y_val, y_val_predict[:, 1])
        # scores.append(fold_auc)
    return scores


