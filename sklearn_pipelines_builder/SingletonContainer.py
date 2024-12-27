class SingleContainer:
    _instance = None  # Class attribute to hold the singleton instance

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    # run id for mlflow logging
    mlflow_run_id = None
    original_columns = []
    numeric_features = []
    string_features = []
    null_numeric_features = []
    null_string_features = []
    mean_null_numeric_features = []
    mean_null_string_features = []
    columns_to_drop = []
    all_features = []
    response = None
    test_score = None
    cv_score = None
    scoring = None


