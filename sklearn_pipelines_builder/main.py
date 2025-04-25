import argparse
import gc
import shap
import mlflow
import mlflow.catboost
import os
import shutil
import pandas as pd
from sklearn_pipelines_builder.utils.basic_utils import eval_scores
from typing import Any
from sklearn.model_selection import train_test_split
from torch.distributed.pipelining import pipeline

from sklearn_pipelines_builder.infrastructure.ElementFactory import ElementFactory
from sklearn_pipelines_builder.SingletonContainer import SingleContainer
from sklearn_pipelines_builder.infrastructure.Config import Config
from sklearn_pipelines_builder.utils.basic_utils import load_dataset, convert_str_to_list, get_features
from sklearn_pipelines_builder.utils.collect_info import collect_info
from sklearn_pipelines_builder.utils.log_info import log_info
from sklearn_pipelines_builder.utils.logger import logger, LoggerSingleton
from sklearn_pipelines_builder.transformers.ThresholdingWrapper import ThresholdingWrapper
from sklearn_pipelines_builder.utils.plots import save_scatter_plot

global_config = Config()

def store_datasets(step, X_train, X_test, y_train, y_test, response_col=None):
    """
    Stores train and test datasets as Parquet files for a given pipeline step.

    Parameters:
    - output_folder (str): Directory where the files will be stored.
    - step (int): Current step number in the pipeline.
    - X_train (DataFrame): Transformed training dataset.
    - X_test (DataFrame): Transformed test dataset.
    - y_train (Series or DataFrame): Training labels.
    - y_test (Series or DataFrame): Test labels.
    - response_col (str): Name of the target column.
    """
    if response_col is None:
        response_col = global_config.get('response')
    # Combine X and y into one DataFrame
    df_train = X_train.copy()
    df_train[response_col] = y_train

    df_test = X_test.copy()
    df_test[response_col] = y_test

    # Save as Parquet files
    df_train.to_parquet(os.path.join(global_config.output_folder, f"train_step_{step}.parquet"))
    df_test.to_parquet(os.path.join(global_config.output_folder, f"test_step_{step}.parquet"))

    logger.info(f"Stored train/test datasets for step {step} in {global_config.output_folder}")

def setup_dateset(dataset_config={}):
    dataset_file = dataset_config.get('file_name')
    dataset_folder = dataset_config.get('folder')
    if dataset_folder is not None:
        dataset_file = os.path.join(dataset_folder, dataset_file)
    dataset = load_dataset(dataset_file)
    pipeline_steps = dataset_config.get('pipe_line_steps', [])
    for n, pipe_line_config in enumerate(pipeline_steps):
        element_name = pipe_line_config.get('element_type')
        logger.info("%s, %s, %s", n, element_name, global_config._config['output_folder'])
        logger.info("Preparing to run pipe_line %s size X_train_transformed=%s", element_name, len(dataset))
        pipeline = ElementFactory().create_pipe_line(pipe_line_config)
        dataset = pipeline.transform(dataset)
    return dataset


def run_pipeline():
    """
    Execute the pipeline defined in the configuration file.

    Parameters:
    - config_file (str): Path to the YAML configuration file.

    Returns:
    - dict: Execution results including metrics and output file path.
    """

    # Create output folders
    os.makedirs(os.path.join(global_config.output_folder, global_config.run_name), exist_ok=True)

    # Load pipeline steps and data
    pipe_line_steps_config = global_config.get('pipe_line_steps')
    final_step_config = global_config.get('final_step')
    response = global_config.get('response')

    df_train = setup_dateset(global_config.get('train_set'))
    df_submission = setup_dateset(global_config.get('submission_set'))
    df_test = setup_dateset(global_config.get('test_set'))

    collect_info(df_train)

    # df_train = df_train.dropna(subset=['sales']).reset_index(drop=True)
    # df_train = df_train[pd.to_datetime(df_train['date']) > pd.to_datetime('2023-01-01')].reset_index(drop=True)

    y = df_train[response]
    X_train_transformed = df_train.drop(columns=[response])

    y_test = df_test[response]
    X_test_transformed = df_test.drop(columns=[response])

    del df_train, df_test
    gc.collect()

    # if global_config.get('test_set') is not None:
    #     df_test = load_dataset(global_config.get('test_set'))
    #     X = X_all.copy(True)
    #     y = y_all.copy(True)
    #     X_test = df_test.drop(columns=[response])
    #     y_test = df_test[response]
    # else:
    #     from sklearn_pipelines_builder.data_splitters.DateBasedCV import DateBasedCV
    #     date_based_cv = DateBasedCV({'date_column': 'date',
    #                                 'threshold_dates': ['2024-05-01']})
    #     train_idx, test_idx = date_based_cv.split(df_train)[0]
    #     X = X_all.iloc[train_idx].reset_index(drop=True)
    #     y = y_all.iloc[train_idx].reset_index(drop=True)
    #     X_test = X_all.iloc[test_idx].reset_index(drop=True)
    #     y_test = y_all.iloc[test_idx].reset_index(drop=True)


    df_submission_transformed = df_submission
    weight_column = global_config.get('weight_column', None)
    if weight_column is None:
        weight_column = 'weight'
        X_train_transformed[weight_column] = 1
        X_test_transformed[weight_column] = 1
        df_submission[weight_column] = 1
        global_config.set('weight_column', weight_column)
        SingleContainer.meta_training_columns = SingleContainer.meta_training_columns + [weight_column]

    # y = y.copy(True).reset_index(drop=True)
    # y_test = y_test.copy(True).reset_index(drop=True)

    log_info('_initial')
    SingleContainer.meta_training_columns = list(set(SingleContainer.meta_training_columns+['response_copy']))
    # Process pipeline steps
    for n, pipe_line_config in enumerate(pipe_line_steps_config):
        element_name = pipe_line_config.get('element_type')
        logger.info("%s, %s, %s", n, element_name, global_config._config['output_folder'])
        logger.info("Preparing to run pipe_line %s size X_train_transformed=%s", element_name, len(X_train_transformed))
        pipeline = ElementFactory().create_pipe_line(pipe_line_config)

        X_train_transformed['response_copy'] = y
        X_train_transformed = pipeline.fit_transform(X_train_transformed, y)
        y = X_train_transformed.pop('response_copy')

        X_test_transformed['response_copy'] = y_test
        X_test_transformed = pipeline.transform(X_test_transformed)
        y_test = X_test_transformed.pop('response_copy')

        df_submission_transformed = pipeline.transform(df_submission_transformed)
        collect_info(X_train_transformed)
        logger.info("Done element number %s --- %s --- Size of train set %s, %s", n, element_name,
                    len(X_train_transformed), len(y))
        logger.info("Length of submission: %s",len(df_submission_transformed))

        if global_config.get("store_every_step"):
            store_datasets(n, X_train_transformed, X_test_transformed, y, y_test)

    log_info('_final')

    # Final step
    final_step = ElementFactory().create_pipe_line(final_step_config)
    X_train_transformed = final_step.fit_transform(X_train_transformed, y)
    X_test_transformed = final_step.transform(X_test_transformed)


    if global_config.get('prediction_type') == 'class':
        final_step = ThresholdingWrapper(final_step, threshold=0.5)

    weight_train = X_train_transformed[weight_column]
    weight_val = X_test_transformed[weight_column]

    training_scores = eval_scores(X_train_transformed, y, weight_train, final_step)
    validation_scores = eval_scores( X_test_transformed, y_test, weight_val, final_step)

    for score_type in training_scores.keys():
        logger.info('Train score %s = %s', score_type, str(training_scores[score_type]))
        logger.info('Validation score %s = %s', score_type, str(validation_scores[score_type]))

    y_train_pred = final_step.predict(X_train_transformed)
    y_test_pred = final_step.predict(X_test_transformed)

    save_scatter_plot(y_test, y_test_pred, global_config.output_folder, name='test',
                      filename="test_predict_vs_truth.png")
    save_scatter_plot(y, y_train_pred, global_config.output_folder, name='train',
                      filename="train_predict_vs_truth.png")

    X_train_transformed['Predicted'] = y_train_pred
    X_test_transformed['Predicted'] = y_test_pred

    store_datasets(n+1, X_train_transformed, X_test_transformed, y, y_test)

    # Prepare submission
    df_submission_transformed[SingleContainer.response] = final_step.predict(df_submission_transformed)
    required_output_columns = convert_str_to_list(global_config.get('output_columns'))
    missing_columns = list(set(required_output_columns) - (set(df_submission_transformed.columns)))
    if missing_columns:
        df_submission_transformed = pd.concat([df_submission_transformed, df_submission[missing_columns]], axis=1)

    output_file = global_config.get('output_file')
    df_submission_transformed[required_output_columns + [SingleContainer.response]].to_csv(
        os.path.join(global_config.output_folder, output_file), index=False)
    # Need to debug this
    calc_shap_values(X_train_transformed, 'shap_values_train.parquet', final_step)
    calc_shap_values(X_test_transformed, 'shap_values_test.parquet', final_step)



def calc_shap_values(dataset, file_name, final_step):
    datetime_cols = dataset.select_dtypes(include=["datetime64"]).columns.tolist()
    dataset[datetime_cols] = dataset[datetime_cols].astype("int64") // 10 ** 9
    # explainer = shap.Explainer(self.model.model, X[best_features], feature_perturbation="tree_path_dependent")
    explainer = shap.TreeExplainer(final_step[0].model, feature_perturbation="tree_path_dependent")
    shap_values = explainer(dataset[SingleContainer.final_features]).values  # (num_samples, num_features)
    # ✅ Convert to DataFrame
    shap_df = pd.DataFrame(shap_values, columns=SingleContainer.final_features)
    shap_df.to_parquet(os.path.join(global_config.output_folder, file_name), index=False)


def main(config_file: str, config_override=None):
    """
    Main function to start an MLflow run and execute the pipeline.

    Parameters:
    - config_file (str): Path to the YAML configuration file.
    """
    global_config.load_config(config_file, config_override)
    os.makedirs(global_config.output_folder, exist_ok=True)
    LoggerSingleton.update_log_file('sklearn_pipelines_builder',
                                    os.path.join(global_config.output_folder, 'run_log.txt'))

    with mlflow.start_run() as run:
        shutil.copy(config_file, global_config.output_folder)
        print("MLflow Tracking URI:", mlflow.get_tracking_uri())


        mlflow.set_experiment(global_config.get('experiment_name'))
        mlflow.set_tag("mlflow.runName", global_config.run_name)
        SingleContainer.mlflow_run_id = run.info.run_id
        mlflow.log_param('ConfigFile', config_file)

        run_pipeline()
        logger.info("Pipeline executed successfully.")


if __name__ == "__main__":
    # Set up the argument parser
    parser = argparse.ArgumentParser(description="Process a YAML configuration file.")
    parser.add_argument(
        "-config_path",
        type=str,
        help="Path to the YAML configuration file.",
        required=False,
        default=r"C:\Projects\sklearn_pipelines_builder_github\tests\rohlik_sales\rohlik-sales-20_75_feature_selector_nn.yaml"
    )

    # Parse the arguments
    args = parser.parse_args()

    # Run the main function with the loaded config
    main(args.config_path)
