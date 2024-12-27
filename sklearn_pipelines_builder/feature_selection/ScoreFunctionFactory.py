from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.feature_selection import f_classif, f_regression
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression


class ScoreFunctionFactory:
    """Factory to generate score functions for SelectKBest."""

    def __init__(self):
        # Mapping of score_func_name to corresponding implementations
        self.score_functions = {
            "decision_tree_regressor": self._decision_tree_regressor_importance,
            "decision_tree_classifier": self._decision_tree_classifier_importance,
            "f_score_regression": f_regression,
            "f_score_classification": f_classif,
            "mutual_info_regression": mutual_info_regression,
            "mutual_info_classification": mutual_info_classif,
        }

    def create(self, score_func_name, **kwargs):
        """
        Create a score function based on the name.

        Parameters:
        - score_func_name (str): The name of the score function.
        - kwargs: Additional parameters for the score function.

        Returns:
        - Callable: A score function compatible with SelectKBest.
        """
        if score_func_name not in self.score_functions:
            raise ValueError(f"Unsupported score function: {score_func_name}")

        score_func = self.score_functions[score_func_name]
        # If the score function requires custom logic (e.g., tree-based), wrap it
        if callable(score_func):
            return lambda X, y: score_func(X, y, **kwargs)
        return score_func

    def _decision_tree_regressor_importance(self, X, y, **kwargs):
        """
        Custom score function based on feature importance from DecisionTreeRegressor.
        """
        tree = DecisionTreeRegressor(random_state=42, **kwargs)
        tree.fit(X, y)
        return tree.feature_importances_

    def _decision_tree_classifier_importance(self, X, y, **kwargs):
        """
        Custom score function based on feature importance from DecisionTreeClassifier.
        """
        tree = DecisionTreeClassifier(random_state=42, **kwargs)
        tree.fit(X, y)
        return tree.feature_importances_

score_fun = ScoreFunctionFactory().create('mutual_info_classification')
