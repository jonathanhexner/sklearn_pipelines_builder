# base_feature_selector.py
import os
import json
import copy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mlflow
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn_pipelines_builder.infrastructure.Config import Config
from sklearn_pipelines_builder.infrastructure.ElementFactory import ElementFactory
from sklearn_pipelines_builder.SingletonContainer import SingleContainer
from sklearn_pipelines_builder.utils.logger import logger
from sklearn_pipelines_builder.utils.basic_utils import eval_scores
from sklearn_pipelines_builder.utils.custom_scorer import get_custom_scorer
from sklearn_pipelines_builder.data_splitters.create_splitter import create_splitter
from sklearn_pipelines_builder.utils.plots import save_scatter_plot


class BaseFeatureSelector(BaseEstimator, TransformerMixin):
    def __init__(self, config):
        self.config = config
        self.model_transformer = ElementFactory().create(config.get("model_config", {}))
        self.cv_config = config.get("cv", {})
        self.train_data_split_config = config.get("train_data_split", {})
        self.meta_training_columns = SingleContainer.meta_training_columns
        self.scoring = get_custom_scorer(config.get("scoring", Config().scoring))
        self.weight_column = Config().get("weight_column", None)
        self.output_folder = config.get("output_folder", Config().output_folder)
        self.response = SingleContainer.response
        os.makedirs(self.output_folder, exist_ok=True)

        self.cv_splitter = create_splitter(self.cv_config)
        self.train_data_splitter = None
        if self.train_data_split_config:
            self.train_data_splitter = create_splitter(self.train_data_split_config)

        self.results_ = []
        self.selected_features_ = None
        self.sorted_features = []
        self.all_features = []

    @property
    def model(self):
        return self.model_transformer

    def _cross_val_score(self, X, y, selected_features):
        val_scores = []
        train_scores = []
        for n_split, (train_idx, val_idx) in enumerate(self.cv_splitter.split(X, y)):
            X_train, X_val = (X.loc[train_idx, selected_features + self.meta_training_columns],
                              X.loc[val_idx, selected_features + self.meta_training_columns])
            y_train, y_val = y.loc[train_idx], y.loc[val_idx]
            self.model_transformer.fit(X_train, y_train)
            y_train_pred = self.model_transformer.predict(X_train)
            y_val_pred = self.model_transformer.predict(X_val)

            sample_weights_train = X_train[self.weight_column] if self.weight_column else None
            sample_weights_val = X_val[self.weight_column] if self.weight_column else None

            val_scores.append(self.scoring._score_func(y_val, y_val_pred, sample_weight=sample_weights_val))
            train_scores.append(self.scoring._score_func(y_train, y_train_pred, sample_weight=sample_weights_train))

            training_scores = eval_scores(X_train, y_train, sample_weights_train, self.model_transformer)
            validation_scores = eval_scores(X_val, y_val, sample_weights_val, self.model_transformer)

            for score_type in training_scores.keys():
                logger.info('Train score %s = %s', score_type, str(training_scores[score_type]))
                logger.info('Validation score %s = %s', score_type, str(validation_scores[score_type]))

        return np.mean(val_scores), np.mean(train_scores)

    def store_features_selected(self):
        json_file = os.path.join(self.output_folder, "selected_features.json")
        with open(json_file, "w") as f:
            json.dump(self.selected_features_, f)
        mlflow.log_artifact(json_file)

    def update_feature_importance_order(self, feature_names):
        logger.info('updating feature importance')
        importances = self.model_transformer.get_feature_importance()
        feature_importance_dict = dict(zip(feature_names, importances))
        missing_features = list(set(self.all_features) - set(feature_names))
        feature_importance_dict.update(dict(zip(missing_features, [-10] * len(missing_features))))
        self.sorted_features = sorted(feature_importance_dict, key=feature_importance_dict.get)

    def transform(self, X):
        if self.selected_features_ is None:
            raise ValueError("The transformer has not been fitted yet.")
        return X[[col for col in X.columns if col in self.selected_features_ + self.meta_training_columns]]

    def predict(self, X):
        if self.selected_features_ is None:
            raise ValueError("The transformer has not been fitted yet.")
        return self.model_transformer.predict(X[self.selected_features_ + self.meta_training_columns])

    def _plot_results(self, X, y):
        selected_X = X[self.selected_features_]
        for n_split, (train_idx, val_idx) in enumerate(self.cv_splitter.split(X, y)):
            X_train, X_val = selected_X.loc[train_idx], selected_X.loc[val_idx]
            y_train, y_val = y.loc[train_idx], y.loc[val_idx]
            y_train_pred = self.model_transformer.predict(X_train)
            y_val_pred = self.model_transformer.predict(X_val)
            X_train['Predict'] = y_train_pred
            X_val['Predict'] = y_val_pred
            save_scatter_plot(y_train, y_train_pred, self.output_folder, name='train', filename="train_pred_vs_actual.png")
            save_scatter_plot(y_val, y_val_pred, self.output_folder, name='validation', filename="val_pred_vs_actual.png")
            break

    def _plot_feature_scores(self):
        if not self.results_:
            return
        num_features, scores = zip(*self.results_)
        plt.figure(figsize=(8, 5))
        plt.scatter(num_features, scores, marker='o', linestyle='-', color='b')
        plt.xlabel("Number of Features")
        plt.ylabel("Validation Score")
        plt.title("Feature Selection Progress")
        plt.gca().invert_xaxis()
        plot_path = os.path.join(self.output_folder, "feature_selection_plot.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        logger.info(f"Feature selection plot saved to {plot_path}")
        plt.close()


    def _plot_feature_importances(self):
        if not hasattr(self.model_transformer, "get_feature_importance") or self.selected_features_ is None:
            logger.warning("Model does not support feature importances or no features were selected.")
            return

        importances = self.model_transformer.get_feature_importance()
        feature_importance_dict = dict(zip(self.selected_features_, importances))
        df = pd.DataFrame({"Feature": self.selected_features_, "Importance": importances})
        csv_path = os.path.join(self.output_folder, "feature_importances.csv")
        df.to_csv(csv_path, index=False)
        mlflow.log_artifact(csv_path)

        sorted_items = sorted(feature_importance_dict.items(), key=lambda x: x[1], reverse=True)
        plt.figure(figsize=(20, 6))
        plt.bar([k for k, _ in sorted_items], [v for _, v in sorted_items])
        plt.xticks(rotation=90)
        plt.xlabel("Features")
        plt.ylabel("Importance")
        plt.title("Feature Importances")

        plot_path = os.path.join(self.output_folder, "feature_importances.png")
        plt.savefig(plot_path, bbox_inches='tight')
        mlflow.log_artifact(plot_path)
        plt.close()
        logger.info(f"Feature importances saved to {plot_path} and logged to MLflow")
