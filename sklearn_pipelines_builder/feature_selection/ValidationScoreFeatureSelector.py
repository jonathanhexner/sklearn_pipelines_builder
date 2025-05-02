import json
import pandas as pd
import numpy as np
from copy import deepcopy
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn_pipelines_builder.utils.basic_utils import get_features

from sklearn_pipelines_builder.infrastructure.ElementFactory import ElementFactory
from sklearn_pipelines_builder.utils.custom_scorer import get_custom_scorer
from sklearn_pipelines_builder.infrastructure.Config import Config
from sklearn_pipelines_builder.utils.logger import logger
from sklearn_pipelines_builder.utils.basic_utils import log_memory
from sklearn_pipelines_builder.feature_selection.BaseFeatureSelector import BaseFeatureSelector
import os


class BaseEvaluationStrategy:
    def evaluate_baseline(self, X, y, sample_weight):
        raise NotImplementedError

    def evaluate_without_feature(self, X, y, feature_name, sample_weight):
        raise NotImplementedError


class RetrainEvaluationStrategy(BaseEvaluationStrategy):
    def __init__(self, model_template, cv, scoring):
        self.model_template = model_template
        self.cv = cv
        self.scoring = scoring

    def evaluate_baseline(self, X, y):
        return self._evaluate(X, y)

    def evaluate_without_feature(self, X, y, feature_name):
        return self._evaluate(X.drop(columns=[feature_name]), y)

    def _evaluate(self, X, y):
        weight_column = Config().get("weight_column", None)
        val_scores = []
        logger.info(f"Evaluating model")
        for n, (train_idx, val_idx) in enumerate(self.cv.split(X, y)):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            sample_weights_train = X_train[weight_column] if weight_column is not None else None
            sample_weights_val = X_val[weight_column] if weight_column is not None else None
            log_memory(f"Before model copy split {n} ")
            model = deepcopy(self.model_template)
            log_memory(f"After model copy split {n} ")
            model.fit(X_train, y_train)
            log_memory(f"After model train split {n}")
            y_val_pred = model.predict(X_val)
            score = self.scoring._score_func(y_val, y_val_pred, sample_weight=sample_weights_val)
            val_scores.append(score)
        logger.info(f"Got validation scores {val_scores}")
        return np.mean(val_scores)


class ShuffleEvaluationStrategy(BaseEvaluationStrategy):
    def __init__(self, model_template, cv, scoring):
        self.model_template = model_template
        self.cv = cv
        self.scoring = scoring
        self.full_model_ = None

    def evaluate_baseline(self, X, y):
        self._train_once(X, y)
        return self._evaluate(X, y, shuffle_feature=None)

    def evaluate_without_feature(self, X, y, feature_name):
        return self._evaluate(X, y, sample_weight)

    def _train_once(self, X, y, sample_weight):
        for train_idx, _ in self.cv.split(X, y):
            X_train = X.iloc[train_idx]
            y_train = y.iloc[train_idx]
            self.full_model_ = deepcopy(self.model_template)
            self.full_model_.fit(X_train, y_train)
            break

    def _evaluate(self, X, y, shuffle_feature=None):
        val_scores = []
        weight_column = Config().get("weight_column", None)

        for fold_idx, (_, val_idx) in enumerate(self.cv.split(X, y)):
            X_val = X.iloc[val_idx]
            y_val = y.iloc[val_idx]
            sample_weights_val = X_val[weight_column] if weight_column is not None else None

            if shuffle_feature is not None:
                X_val[shuffle_feature] = np.random.permutation(X_val[shuffle_feature].values)

            y_val_pred = self.full_model_.predict(X_val)
            score = self.scoring._score_func(y_val, y_val_pred, sample_weight=sample_weights_val)
            val_scores.append(score)

        return np.mean(val_scores)


class EvaluationStrategyFactory:
    @staticmethod
    def create(strategy_name, model_template, cv, scoring):
        if strategy_name == "shuffle":
            return ShuffleEvaluationStrategy(model_template, cv, scoring)
        elif strategy_name == "retrain":
            return RetrainEvaluationStrategy(model_template, cv, scoring)
        else:
            raise ValueError(f"Unknown strategy: {strategy_name}")


class ValidationScoreFeatureSelector(BaseFeatureSelector):
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.cv_config = config.get("cv")
        self.model_config = config.get("model_config", {})
        self.strategy = config.get("strategy", "loo")  # loo or greedy
        self.eval_strategy = config.get("eval_strategy", "retrain")  # retrain or shuffle
        self.cv_mode = config.get("cv_mode", "rolling")  # rolling or fixed_train

        self.output_folder = Config().output_folder
        os.makedirs(self.output_folder, exist_ok=True)

        self.model_factory = ElementFactory()
        self.model_template = self.model_factory.create(self.model_config)
        self.scoring = get_custom_scorer(config.get("scoring", Config().scoring))

        self.selected_features_ = []
        self.dropped_features_ = []
        self.results_ = []


    def _load_summary(self, summary_path):
        if not os.path.exists(summary_path):
            raise FileNotFoundError(f"Feature selection summary file not found: {summary_path}")

        df_summary = pd.read_csv(summary_path)

        selected = df_summary[df_summary['selected'] == True]['feature'].tolist()
        if selected is None:
            raise ValueError("Missing 'selected_features' key in summary file.")

        if not isinstance(selected, list):
            raise TypeError(f"'selected_features' must be a list, got {type(selected)}")

        return selected

    def fit(self, X, y):
        if "feature_selection_summary" in self.config:
            self.selected_features_ = self._load_summary(self.config["feature_selection_summary"])
            return self

        feature_names = get_features(X)

        evaluator = EvaluationStrategyFactory.create(
            strategy_name=self.eval_strategy,
            model_template=self.model_template,
            cv=self.cv_splitter,
            scoring=self.scoring
        )
        logger.info(f"Using evaluation strategy: {self.eval_strategy}")
        log_memory("Before evaluation baseline")
        baseline_score = evaluator.evaluate_baseline(X, y)
        logger.info(f"Baseline validation score with all features: {baseline_score:.5f}")
        log_memory("After evaluation baseline")
        for idx, feature in enumerate(feature_names):
            score = evaluator.evaluate_without_feature(X.drop(columns=self.dropped_features_), y, feature)
            delta = score - baseline_score
            keep = delta < 0
            logger.info(f"Evaluating feature {idx+1}/{len(feature_names)}: {feature} delta score = {delta:.2f}"
                        f"dropped = {not keep}")

            if not keep:
                self.dropped_features_.append(feature)

            self.results_.append({
                "feature": feature,
                "strategy": self.eval_strategy,
                "score_with": baseline_score,
                "score_without": score,
                "score_diff": delta,
                "selected": keep
            })

        self.selected_features_ = [r["feature"] for r in self.results_ if r["selected"]]

        results_df = pd.DataFrame(self.results_)
        results_df.to_csv(os.path.join(self.output_folder, "feature_selection_summary.csv"), index=False)

        logger.info("Feature selection complete. Selected features:")
        logger.info(self.selected_features_)

        return self

    def transform(self, X):
        return X.drop(columns=self.dropped_features_)
