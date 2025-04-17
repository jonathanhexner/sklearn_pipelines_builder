import copy
import os
import numpy as np
import pandas as pd
import shap
import json
import matplotlib
matplotlib.use("agg")  # ✅ Switch to non-interactive backend
import matplotlib.pyplot as plt

import mlflow
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn_pipelines_builder.infrastructure.Config import Config
from sklearn_pipelines_builder.infrastructure.ElementFactory import ElementFactory
from sklearn_pipelines_builder.utils.logger import logger
from sklearn_pipelines_builder.utils.basic_utils import eval_scores

from sklearn_pipelines_builder.data_splitters.create_splitter import create_splitter
from sklearn_pipelines_builder.utils.custom_scorer import get_custom_scorer
from sklearn_pipelines_builder.utils.plots import save_scatter_plot
from sklearn_pipelines_builder.SingletonContainer import SingleContainer

global_config = Config()


class RFECVTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, config):
        """
        Custom RFE transformer that eliminates features based on importance without sklearn validation.

        Parameters:
        - config (dict): Configuration dictionary containing:
            - "model_config": Model to use for feature elimination.
            - "cv": Custom cross-validation strategy.
            - "scoring": Scoring metric for evaluation.
            - "threshold": Maximum allowed performance drop for feature elimination.
            - "output_folder": Path to save plots and artifacts.
        """
        self.config = config
        self.model_transformer = ElementFactory().create(config.get("model_config", {}))
        self.cv_config = config.get("cv", {})
        self.train_data_split_config = config.get("train_data_split", {})
        self.meta_training_columns = SingleContainer.meta_training_columns
        self.max_iterations = config.get('max_iterations', 5)
        self.scoring = get_custom_scorer(config.get("scoring", global_config.scoring))
        self.weight_column = global_config.get('weight_column', None)
        self.min_features = config.get('min_features', 4)
        self.threshold = config.get("threshold", 0.01)  # Allowed performance drop
        self.output_folder = config.get("output_folder", global_config.output_folder)
        self.response = SingleContainer.response
        os.makedirs(self.output_folder, exist_ok=True)
        self.results_ = []
        # Create CV splitter
        self.cv_splitter = create_splitter(self.cv_config)
        self.train_data_splitter = None
        self.sorted_features = []
        self.all_features = []
        if self.train_data_split_config:
            self.train_data_splitter = create_splitter(self.train_data_split_config)

        # Store selected features
        self.selected_features_ = None

    @property
    def model(self):
        return self.model_transformer

    def _cross_val_score(self, X, y, selected_features):
        """
        Perform cross-validation with the selected feature subset.

        Returns:
        - Mean cross-validation score.
        """
        val_scores = []
        train_scores = []
        sample_weights_train = None
        sample_weights_val = None
        for n_split, (train_idx, val_idx) in enumerate(self.cv_splitter.split(X, y)):
            X_train, X_val = (X.loc[train_idx, selected_features+SingleContainer.meta_training_columns],
                              X.loc[val_idx, selected_features+SingleContainer.meta_training_columns])
            y_train, y_val = y.loc[train_idx], y.loc[val_idx]
            # if 'date' in X_train.columns:
            #     X_train.drop(columns=['date'], inplace=True)
            #     X_val.drop(columns=['date'], inplace=True)
            logger.info('Fitting model for split %s', str(n_split))
            self.model_transformer.fit(X_train, y_train)
            y_train_pred = self.model_transformer.predict(X_train)
            y_val_pred = self.model_transformer.predict(X_val)
            if self.weight_column is not None:
                sample_weights_train = X_train[self.weight_column]
                sample_weights_val = X_val[self.weight_column]
            val_scores.append(self.scoring._score_func(y_val, y_val_pred, sample_weight = sample_weights_val))
            train_scores.append(self.scoring._score_func(y_train, y_train_pred, sample_weight = sample_weights_train))

            training_scores = eval_scores(X_train, y_train, sample_weights_train, self.model_transformer)
            validation_scores = eval_scores(X_val, y_val, sample_weights_val, self.model_transformer)

            for score_type in training_scores.keys():
                logger.info('Train score %s = %s', score_type, str(training_scores[score_type]))
                logger.info('Validation score %s = %s', score_type, str(validation_scores[score_type]))

        return np.mean(val_scores), np.mean(train_scores)

    def fit(self, X_input, y_input):
        """
        Perform feature elimination manually without sklearn validation.

        Parameters:
        - X (pd.DataFrame): Input features.
        - y (pd.Series): Target variable.

        Returns:
        - self
        """
        if self.train_data_splitter is not None:
            train_test_idx = self.train_data_splitter.split(X_input, y_input)
            subset_idx = train_test_idx[0][1]
            X = X_input.loc[subset_idx]
            y = y_input.loc[subset_idx]
            weight_column = X_input.loc[subset_idx, self.weight_column]
        else:
            X = X_input.copy()  # Ensure we don't modify the original dataset
            y = y_input.copy()  # Ensure we don't modify the original dataset
            weight_column = X_input.loc[:, self.weight_column].copy(True)
        feature_names = [col for col in X.columns if col not in
                         SingleContainer.meta_training_columns]
        num_features = len(feature_names)
        max_features = num_features
        prev_max_features = max_features
        # Train model once and store importances
        logger.info(f"Preparing to run initial fit using all features %s", str(num_features))

        grand_best_score, train_score = self._cross_val_score(X, y, feature_names)
        best_score = grand_best_score
        best_model = copy.deepcopy(self.model_transformer)

        logger.info(f"Initial validation score %s train score %s", str(best_score), str(train_score))
        self.all_features = copy.deepcopy(feature_names)
        self.update_feature_importance_order(feature_names)
        last_update_iteration = 0
        iteration = 0
        min_features = self.min_features
        best_features = copy.deepcopy(self.sorted_features)
        while (max_features - min_features) > 1:
            if iteration >= self.max_iterations:
                logger.info(f"Reached max iterations: {self.max_iterations}. Stopping early.")
                break

            step_size = (max_features - min_features) // 2
            mid_features = min_features + step_size
            selected_features = self.sorted_features[-mid_features:]

            score, train_score = self._cross_val_score(X, y, selected_features)
            self.results_.append((mid_features, score))

            score_change = ((grand_best_score - score) / abs(grand_best_score))
            logger.info(f"Iteration {iteration}: {mid_features} features, Validation Score: {score: .3f}; "
                        f"Train Score: {train_score: .3f} Best {grand_best_score: .3f}; Score Change {score_change: .3f}")

            if  score_change > self.threshold:
                logger.info(f"Significant performance drop detected ({score_change:.3f}). Restoring previous best.")
                min_features = mid_features  # Restore search range correctly
                max_features = prev_max_features
            else:
                self.update_feature_importance_order(feature_names)
                grand_best_score = max(score, grand_best_score)
                best_features = selected_features
                best_score = score
                best_model = copy.deepcopy(self.model_transformer)
                max_features = mid_features  # Continue binary search
                min_features = max(min_features-2, self.min_features)
            prev_max_features = max_features
            iteration += 1

        logger.info(f"Final selected feature count: {len(best_features)} with score {best_score:.4f}")

        self.selected_features_ = best_features
        self.model_transformer = best_model
        logger.info(f"Feature selection completed. Selected {len(best_features)} features.")
        self._plot_feature_scores()
        self._plot_results(X, y)

        # Log performance to MLflow
        mlflow.log_metric("rfe_optimal_cv_score", best_score)

        # Plot feature importances
        self._plot_feature_importances()
        self.store_features_selected()
        SingleContainer.final_features = self.selected_features_
        # self.store_shap_values(X)
        X[self.weight_column] = weight_column
        X[self.response] =y
        X.to_parquet(os.path.join(self.output_folder, "RFECV_train.parquet"), index=False)
        return self

    def store_features_selected(self):
        # Save as JSON
        json_file = os.path.join(global_config.output_folder, "selected_features.json")
        with open(json_file, "w") as f:
            json.dump(self.selected_features_, f)

        # Log as artifact
        mlflow.log_artifact(json_file)

    def store_shap_values(self, X):
        explainer = shap.Explainer(self.model_transformer.model, X)
        shap_values = explainer(X).values  # (num_samples, num_features)
        # ✅ Convert to DataFrame
        shap_df = pd.DataFrame(shap_values, columns=X.columns)
        shap_df.to_csv(os.path.join(self.output_folder, "shap_values.csv"), index=False)

    def update_feature_importance_order(self, feature_names):
        logger.info('updating feature importance')
        importances = self.model_transformer.get_feature_importance()
        feature_importance_dict = dict(zip(feature_names, importances))
        missing_features = list(set(self.all_features)-set(feature_names))
        feature_importance_dict.update(dict(zip(missing_features, [-10]*len(missing_features))))
        self.sorted_features = sorted(feature_importance_dict, key=feature_importance_dict.get)

    def predict(self, X):
        """
        Transform the dataset by keeping only the selected features.

        Parameters:
        - X (pd.DataFrame): Input features.

        Returns:
        - Transformed DataFrame with selected features.
        """
        if self.selected_features_ is None:
            raise ValueError("The transformer has not been fitted yet.")

        return self.model_transformer.predict(X[self.selected_features_+self.meta_training_columns])

    def transform(self, X):
        """
        Transform the dataset by keeping only the selected features.

        Parameters:
        - X (pd.DataFrame): Input features.

        Returns:
        - Transformed DataFrame with selected features.
        """
        if self.selected_features_ is None:
            raise ValueError("The transformer has not been fitted yet.")
        subset_cols = [col for col in X.columns if col in
                       self.selected_features_+self.meta_training_columns]
        return X[subset_cols]

    def evaluate_final_model(self, X_train, X_test, y_train, y_test):
        """
        Evaluate the final model using the selected features.

        Parameters:
        - X_train, X_test: Train and test features.
        - y_train, y_test: Train and test targets.

        Returns:
        - final_score (float): The final model evaluation score.
        """
        if self.selected_features_ is None:
            raise ValueError("No features selected. Run `fit` first.")

        # Fit the model on selected features
        self.model_transformer.fit(X_train[self.selected_features_], y_train)

        # Predict on test set
        y_pred = self.model_transformer.predict(X_test[self.selected_features_])

        # Calculate and log the final score
        final_score = self.scoring._score_func(y_test, y_pred)
        mlflow.log_metric("final_model_score", final_score)

        return final_score

    def _plot_results(self, X, y):
        selected_X = X[self.selected_features_]
        for n_split, (train_idx, val_idx) in enumerate(self.cv_splitter.split(X, y)):
            X_train, X_val = selected_X.loc[train_idx], selected_X.loc[val_idx]
            y_train, y_val = y.loc[train_idx], y.loc[val_idx]
            y_train_pred = self.model_transformer.predict(X_train)
            y_val_pred = self.model_transformer.predict(X_val)
            X_train['Predict'] = y_train_pred
            X_val['Predict'] = y_val_pred

            save_scatter_plot(y_train, y_train_pred, global_config.output_folder, name='RFECV_train',
                              filename="train_predict_vs_actual.png")
            save_scatter_plot(y_val, y_val_pred, global_config.output_folder, name='RFECV_validation',
                              filename="validation_predict_vs_actual.png")
            # Save as Parquet files
            X_train.to_parquet(os.path.join(global_config.output_folder, f"train_RFECV.parquet"))
            X_val.to_parquet(os.path.join(global_config.output_folder, f"test_RFECV.parquet"))

            if n_split>0:
                break

    def _plot_feature_scores(self):
        """Plots feature count vs validation score and saves to a file."""
        if not self.results_:
            return

        num_features, scores = zip(*self.results_)
        plt.figure(figsize=(8, 5))
        plt.scatter(num_features, scores, marker='o', linestyle='-', color='b')
        plt.xlabel("Number of Features")
        plt.ylabel("Validation Score")
        plt.title("Feature Selection Progress")
        plt.gca().invert_xaxis()  # More features on the left
        plt.grid(True)

        # Save plot
        plot_path = os.path.join(self.output_folder, "feature_selection_plot.png")

        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        logger.info(f"Feature selection plot saved to {plot_path}")
        plt.close()  # Close to prevent display in notebooks


    def _plot_feature_importances(self):
        """
        Plot and save feature importances based on the final selected model.
        """
        if not hasattr(self.model_transformer, "get_feature_importance"):
            return

        feature_importance_dict = self._get_model_feature_importances()
        df_feature_importances = pd.DataFrame({'Feature': list(feature_importance_dict.keys()),
                                               'Value': list(feature_importance_dict.values())})
        df_feature_importances.to_csv(os.path.join(self.output_folder, 'df_feature_importance.csv'), index=False)
        sorted_importances = sorted(feature_importance_dict.items(), key=lambda x: x[1], reverse=True)

        plt.figure(figsize=(20, 6))
        plt.bar([x[0] for x in sorted_importances], [x[1] for x in sorted_importances])
        plt.xlabel("Features")
        plt.ylabel("Importance")
        plt.title("Feature Importances from Custom RFE")

        plot_path = os.path.join(self.output_folder, "feature_importances.png")
        plt.savefig(plot_path)
        plt.close()

        # Log the plot to MLflow
        mlflow.log_artifact(plot_path)

    def _get_model_feature_importances(self):
        importances = self.model_transformer.get_feature_importance()
        feature_importance_dict = dict(zip(self.selected_features_, importances))
        return feature_importance_dict
