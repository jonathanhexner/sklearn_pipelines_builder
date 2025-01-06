import argparse
import mlflow
import mlflow.catboost
import os
import pandas as pd
from typing import Any
from sklearn.model_selection import train_test_split
from sklearn_pipelines_builder.infrastructure.ElementFactory import ElementFactory
from sklearn_pipelines_builder.SingletonContainer import SingleContainer
from sklearn_pipelines_builder.infrastructure.Config import Config
from sklearn_pipelines_builder.utils.basic_utils import load_dataset, convert_str_to_list
from sklearn_pipelines_builder.utils.collect_info import collect_info
from sklearn_pipelines_builder.utils.log_info import log_info
from sklearn_pipelines_builder.utils.logger import logger
from sklearn_pipelines_builder.transformers.ThresholdingWrapper import ThresholdingWrapper

from sklearn.metrics import get_scorer

config = Config()


def run_pipeline():
    """
    Execute the pipeline defined in the configuration file.

    Parameters:
    - config_file (str): Path to the YAML configuration file.

    Returns:
    - dict: Execution results including metrics and output file path.
    """

    # Create output folders
    os.makedirs(config.output_folder, exist_ok=True)
    os.makedirs(os.path.join(config.output_folder, config.run_name), exist_ok=True)

    # Load pipeline steps and data
    pipe_line_steps_config = config.get('pipe_line_steps')
    final_step_config = config.get('final_step')
    response = config.get('response')

    df_train = load_dataset(config.get('train_set'))
    df_submission = load_dataset(config.get('test_set'))
    collect_info(df_train)

    y_all = df_train[response]
    X_all = df_train.drop(columns=[response])

    X, X_test, y, y_test = train_test_split(
        X_all, y_all, test_size=0.2, random_state=42, stratify=y_all
    )

    # Prepare transformed datasets
    X_train_transformed = X.copy(True).reset_index(drop=True)
    X_test_transformed = X_test.copy(True).reset_index(drop=True)
    df_submission_transformed = df_submission.copy(True).reset_index(drop=True)
    y = y.copy(True).reset_index(drop=True)
    y_test = y_test.copy(True).reset_index(drop=True)

    collect_info(X)
    log_info('_initial')

    # Process pipeline steps
    for n, pipe_line_config in enumerate(pipe_line_steps_config):
        element_name = pipe_line_config.get('element_name')
        logger.info(n, element_name, config._config['output_folder'])
        logger.info("Preparing to run pipe_line %s size X=%s", element_name, len(X))
        pipeline = ElementFactory().create_pipe_line(pipe_line_config)
        X_train_transformed = pipeline.fit_transform(X_train_transformed, y)
        X_test_transformed = pipeline.transform(X_test_transformed)
        df_submission_transformed = pipeline.transform(df_submission_transformed)
        collect_info(X_train_transformed)
        logger.info("Length of submission: %s",len(df_submission_transformed))

    log_info('_final')

    # Final step
    final_step = ElementFactory().create_pipe_line(final_step_config)
    X_train_transformed = final_step.fit_transform(X_train_transformed, y)
    scorer = get_scorer(Config().scoring)
    thresholded_model = ThresholdingWrapper(final_step, threshold=0.5)
    test_score = scorer(thresholded_model, X_test_transformed, y_test)

    # Prepare submission
    df_submission_transformed[SingleContainer.response] = thresholded_model.predict(df_submission_transformed)
    required_output_columns = convert_str_to_list(config.get('output_columns'))
    missing_columns = list(set(required_output_columns) - (set(df_submission_transformed.columns)))
    if missing_columns:
        df_submission_transformed = pd.concat([df_submission_transformed, df_submission[missing_columns]], axis=1)

    output_file = config.get('output_file')
    df_submission_transformed[required_output_columns + [SingleContainer.response]].to_csv(output_file, index=False)

    return {
        "test_score": test_score,
        "output_file": output_file
    }



def main(config_file: str, config_override=None):
    """
    Main function to start an MLflow run and execute the pipeline.

    Parameters:
    - config_file (str): Path to the YAML configuration file.
    """
    config.load_config(config_file, config_override)
    with mlflow.start_run() as run:
        mlflow.set_experiment(config.get('experiment_name'))
        mlflow.set_tag("mlflow.runName", config.run_name)
        SingleContainer.mlflow_run_id = run.info.run_id
        mlflow.log_param('ConfigFile', config_file)

        results = run_pipeline()
        mlflow.log_metric('test_score', results["test_score"])
        logger.info("Pipeline executed successfully. Results: %s", results)


if __name__ == "__main__":
    # Set up the argument parser
    parser = argparse.ArgumentParser(description="Process a YAML configuration file.")
    parser.add_argument(
        "-config_path",
        type=str,
        help="Path to the YAML configuration file.",
        required=False,
        default=r"C:\Projects\sklearn_pipelines_builder_github\tests\kaggle_s4e11\config_custom_transformer_featurewiz.yaml"
    )

    # Parse the arguments
    args = parser.parse_args()

    # Run the main function with the loaded config
    main(args.config_path)
