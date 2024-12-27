import argparse
import copy
import optuna
from optuna.pruners import MedianPruner

import yaml
import mlflow
import mlflow.catboost
import sys
import os
import pandas as pd
from sklearn.model_selection import train_test_split
# from tests.test_selection.conftest import df_test

from sklearn_pipelines_builder.infrastructure.ElementFactory import ElementFactory
from sklearn_pipelines_builder.factories.MultiColumnElementFactory import MultiColumnElementFactory
from sklearn_pipelines_builder.SingletonContainer import SingleContainer
from sklearn_pipelines_builder.infrastructure.Config import Config
from sklearn_pipelines_builder.utils.basic_utils import load_dataset, convert_str_to_list
from sklearn_pipelines_builder.utils.collect_info import collect_info
from sklearn_pipelines_builder.utils.log_info import log_info
from sklearn_pipelines_builder.utils.logger import logger

from sklearn.pipeline import Pipeline
from sklearn.metrics import get_scorer


config = Config()

# def load_config(file_path):
#     """Load configuration from a YAML file."""
#     with open(file_path, 'r') as file:
#         config = yaml.safe_load(file)
#     return config



def main(config_file: str):
    # with mlflow.start_run() as run:
    config.load_config(config_file)
    mlflow.set_tag("mlflow.runName", config.run_name)
    SingleContainer.mlflow_run_id = run.info.run_id
    mlflow.log_param('ConfigFile', config_file)

    # Load the YAML configuration

    os.makedirs(config.output_folder, exist_ok=True)
    os.makedirs(os.path.join(config.output_folder, config.run_name), exist_ok=True)

    pipe_line_steps_config = config.get('pipe_line_steps')
    final_step_config = config.get('final_step')
    all_data_pipe_line_steps_config = config.get('all_data_pipe_lines_steps')

    response = config.get('response')
    df_train = load_dataset(config.get('train_set'))
    df_submission = load_dataset(config.get('test_set'))
    collect_info(df_train)


    transformers = []
    for pipe_line_config in all_data_pipe_line_steps_config:
        transformers.append((pipe_line_config.get('element_name'), MultiColumnElementFactory().create(pipe_line_config)))
    if len(transformers)>0:
        pipeline = Pipeline(transformers)
        df_train = pipeline.fit_transform(df_train, df_train[response])
        df_submission = pipeline.transform(df_submission)
    y_all = df_train[response]
    X_all = df_train.drop(columns=[response])

    X, X_test, y, y_test = train_test_split(
        X_all, y_all, test_size=0.2, random_state=42, stratify=y_all
    )

    X_train_transformed = X.copy(True).reset_index(drop=True)
    X_test_transformed = X_test.copy(True).reset_index(drop=True)
    df_submission_transformed = df_submission.copy(True).reset_index(drop=True)
    y = y.copy(True).reset_index(drop=True)
    y_test = y_test.copy(True).reset_index(drop=True)
    transformers = []
    collect_info(X)
    log_info('_initial')

    for pipe_line_config in pipe_line_steps_config:
        element_name = pipe_line_config.get('element_name')
        logger.info("Preparing to run pipe_line %s size X=%s", element_name, len(X))
        # pipeline = MultiColumnElementFactory().create(pipe_line_config)
        pipeline = ElementFactory().create(pipe_line_config)
        X_train_transformed = pipeline.fit_transform(X_train_transformed, y)
        X_test_transformed = pipeline.transform(X_test_transformed)
        df_submission_transformed = pipeline.transform(df_submission_transformed)
        collect_info(X_train_transformed)

    log_info('_final')

    final_step = ElementFactory().create(final_step_config)
    X_train_transformed = final_step.fit_transform(X_train_transformed, y)
    scorer = get_scorer(Config().scoring)
    test_score  = scorer(final_step, X_test_transformed, y_test)
    mlflow.log_metric('test_score', test_score)

    df_submission_transformed[SingleContainer.response] = final_step.predict(df_submission_transformed)
    required_output_columns = convert_str_to_list(config.get('output_columns'))
    missing_columns = list(set(required_output_columns)-(set(df_submission_transformed.columns)))
    if len(missing_columns)>0:
        df_submission_transformed = pd.concat([df_submission_transformed, df_submission[missing_columns]], axis=1)
    df_submission_transformed[required_output_columns+[SingleContainer.response]].to_csv(config.get('output_file'), index=False)

    # hyper_parameter_optimization = config.hyper_parameter_optimization
    # if hyper_parameter_optimization.get('optimizer')=='optuna':
    #     study = optuna.create_study(direction="maximize", pruner=MedianPruner())
    #     study.optimize(lambda trial: objective_catboost(trial, X_train_transformed, y),
    #                    n_trials=hyper_parameter_optimization.get('n_trials'))
    #     # After study.optimize() is done, retrieve the best hyperparameters
    #     best_trial = study.best_trial
    #     best_params = best_trial.params
    #     best_validation_score = best_trial.value
    #     best_model = best_trial.user_attrs["trained_model"]
    #
    #     scorer = get_scorer(Config().scoring)
    #     test_score  = scorer(best_model, X_test_transformed, y_test)
    #     logger.info("Best hyperparameters: %s", best_params)
    #     mlflow.log_params(best_params)
    #     mlflow.log_metric('best_cv_score', best_validation_score)
    #     mlflow.log_metric('test_score', test_score)
    #     if config.get('prediction_type') in ['class', 'regression']:
    #         df_submission_transformed[SingleContainer.response] = best_model.predict(df_submission_transformed)
    #     else:
    #         df_submission_transformed[SingleContainer.response] = best_model.predict_proba(df_submission_transformed)[:, 1]
    #     required_output_columns = convert_str_to_list(config.get('output_columns'))
    #     missing_columns = list(set(required_output_columns)-(set(df_submission_transformed.columns)))
    #     if len(missing_columns)>0:
    #         df_submission_transformed = pd.concat([df_submission_transformed, df_submission[missing_columns]], axis=1)
    #
    #     df_submission_transformed[required_output_columns+[SingleContainer.response]].to_csv(config.get('output_file'), index=False)
    #
    #     # Optionally, you can also get the best score (objective value) from the best trial
    #     best_score = study.best_value
    #     logger.info("Best %s score: %s", global_config.scoring, best_score)
    # else:
    #     raise ValueError(f"Unsupported optimizer {hyper_parameter_optimization.get('optimizer')}")







if __name__ == "__main__":
    # Set up the argument parser
    parser = argparse.ArgumentParser(description="Process a YAML configuration file.")
    parser.add_argument("-config_path", type=str, help="Path to the YAML configuration file.",
                        required=False,
                        default=r"C:\Projects\sklearn_pipelines_builder\tests\config_custom_transformer_refactor.yml")

    # Parse the arguments
    args = parser.parse_args()

    # Run the main function with the loaded config
    with mlflow.start_run() as run:
        main(args.config_path)

