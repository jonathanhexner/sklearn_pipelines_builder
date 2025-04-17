import copy
from sklearn_pipelines_builder.feature_selection.BaseFeatureSelector import BaseFeatureSelector
from sklearn_pipelines_builder.utils.logger import logger

class HybridSelectorTransformer(BaseFeatureSelector):
    def __init__(self, config):
        super().__init__(config)
        self.k_fixed = config.get("k_fixed", 5)
        self.k_greedy = config.get("k_greedy", 5)
        self.greedy_pool_size = config.get("greedy_pool_size", 30)

    def fit(self, X, y):
        feature_names = [col for col in X.columns if col not in self.meta_training_columns]
        self.all_features = copy.deepcopy(feature_names)
        self.update_feature_importance_order(feature_names)

        fixed_features = self.sorted_features[-self.k_fixed:]
        candidates = self.sorted_features[-(self.k_fixed + self.greedy_pool_size):-self.k_fixed]
        selected = fixed_features.copy()
        remaining = candidates.copy()

        best_score, _ = self._cross_val_score(X, y, selected)
        logger.info(f"Initial score with top-{self.k_fixed}: {best_score:.4f}")

        for i in range(self.k_greedy):
            best_feature = None
            best_new_score = -np.inf

            for feature in remaining:
                trial_features = selected + [feature]
                score, _ = self._cross_val_score(X, y, trial_features)
                if score > best_new_score:
                    best_new_score = score
                    best_feature = feature

            if best_feature:
                selected.append(best_feature)
                remaining.remove(best_feature)
                best_score = best_new_score
                logger.info(f"Added feature {best_feature}, score: {best_score:.4f}")
            else:
                logger.info("No further improvement. Stopping.")
                break

        self.selected_features_ = selected
        self.store_features_selected()
        self._plot_feature_scores()
        self._plot_feature_importances()
        self._plot_results(X, y)
        return self
