from sklearn.model_selection import KFold, StratifiedKFold
from sklearn_pipelines_builder.data_splitters.DateBasedCV import DateBasedCV

def create_splitter(config):
    """
    Factory function to create a CV splitter based on the configuration.

    Parameters:
    - config (dict): Configuration dictionary containing:
        - split_type (str): Type of CV ('date_based', 'kfold', 'stratified').
        - Other parameters specific to the CV type.

    Returns:
    - A scikit-learn compatible splitter.
    """
    split_type = config.get("split_type", "kfold")

    if split_type == "date_based":
        return DateBasedCV(config=config)
    elif split_type == "kfold":
        n_splits = config.get("n_splits", 5)
        return KFold(n_splits=n_splits, shuffle=True, random_state=config.get("random_state", 42))
    elif split_type == "stratified":
        n_splits = config.get("n_splits", 5)
        return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=config.get("random_state", 42))
    else:
        raise ValueError(f"Unsupported CV type: {split_type}")
