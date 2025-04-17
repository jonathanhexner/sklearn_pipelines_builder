import copy
from sklearn_pipelines_builder.feature_selection.BaseFeatureSelector import BaseFeatureSelector
from sklearn_pipelines_builder.utils.logger import logger


class RFECVTransformer(BaseFeatureSelector):
    def __init__(self, config):
        super().__init__(config)
        self.max_iterations = config.get('max_iterations', 5)
        self.min_features = config.get('min_features', 4)
        self.threshold = config.get("threshold", 0.01)

    def fit(self, X, y):
        feature_names = [col for col in X.columns if col not in self.meta_training_columns]
        self.all_features = copy.deepcopy(feature_names)
        self.update_feature_importance_order(feature_names)

        grand_best_score, _ = self._cross_val_score(X, y, feature_names)
        best_features = copy.deepcopy(self.sorted_features)

        iteration = 0
        min_features = self.min_features
        max_features = len(feature_names)
        prev_max_features = max_features

        while (max_features - min_features) > 1:
            if iteration >= self.max_iterations:
                logger.info(f"Reached max iterations: {self.max_iterations}. Stopping early.")
                break

            step_size = (max_features - min_features) // 2
            mid_features = min_features + step_size
            selected_features = self.sorted_features[-mid_features:]

            score, _ = self._cross_val_score(X, y, selected_features)
            self.results_.append((mid_features, score))

            score_change = ((grand_best_score - score) / abs(grand_best_score))
            logger.info(f"Iteration {iteration}: {mid_features} features, Score: {score:.4f}, Change: {score_change:.4f}")

            if score_change > self.threshold:
                min_features = mid_features
                max_features = prev_max_features
            else:
                best_features = selected_features
                grand_best_score = max(score, grand_best_score)
                max_features = mid_features
                min_features = max(min_features - 2, self.min_features)

            prev_max_features = max_features
            iteration += 1

        self.selected_features_ = best_features
        self.store_features_selected()
        self._plot_feature_scores()
        self._plot_feature_importances()
        self._plot_results(X, y)
        return self
