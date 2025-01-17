import sys
import os
import pandas as pd
import unittest
from sklearn_pipelines_builder.utils.test_utils import get_test_folder, get_results_folder
from sklearn_pipelines_builder.utils.basic_utils import create_clean_folder
from sklearn_pipelines_builder.main import main


class TestKaggleS4e10(unittest.TestCase):
    """
    Test suite for executing external modules with existing YAML configurations.
    """

    @classmethod
    def setup_class(cls):
        cls._test_folder = os.path.join(get_test_folder(), 'kaggle_s4e10')
        cls._main_output_folder = os.path.join(get_results_folder(), 'kaggle_s4e10')
        create_clean_folder(cls._main_output_folder)
        cls._train_set = os.path.join(cls._test_folder, 'train.zip')
        cls._test_set = os.path.join(cls._test_folder, 'test.zip')
    def test_autogluon_config(self):
        """
        Test execution with a valid configuration file.
        """
        config_path = os.path.join(self._test_folder, 'config_custom_transformer_autogluon.yaml')
        output_folder = os.path.join(self._main_output_folder, 'autogluon')
        output_file = os.path.join(output_folder, 'submission_file.csv')
        create_clean_folder(output_folder)
        config_override = {'output_folder': output_folder,
                          'train_set': self._train_set,
                          'test_set': self._test_set,
                          'output_file': output_file}
        main(config_path, config_override)

        # Validate the result against expected criteria
        autogluon_file = os.path.join(output_folder, 'config_custom_transformer_autogluon',
                                      'AutoGluonModel.csv')
        self.assertEqual(os.path.exists(autogluon_file), True)
        df_auto_gluon = pd.read_csv(autogluon_file)
        self.assertEqual(len(df_auto_gluon),8)
        self.assertEqual(os.path.exists(output_file), True)
        df_submission = pd.read_csv(output_file)
        self.assertEqual(len(df_submission),39098)

    def test_catboost_optuna_config(self):
        """
        Test execution with a valid configuration file.
        """
        config_path = os.path.join(self._test_folder, 'config_custom_transformer_catboost_optuna.yaml')
        output_folder = os.path.join(self._main_output_folder, 'optuna')
        output_file = os.path.join(output_folder, 'submission_file.csv')
        create_clean_folder(output_folder)
        config_override = {'output_folder': output_folder,
                          'train_set': self._train_set,
                          'test_set': self._test_set,
                          'output_file': output_file}
        main(config_path, config_override)

        df_submission = pd.read_csv(output_file)
        self.assertEqual(len(df_submission),39098)

    def test_feature_wiz_config(self):
        """
        Test execution with a valid configuration file.
        """
        config_path = os.path.join(self._test_folder, 'config_custom_transformer_featurewiz.yaml')
        output_folder = os.path.join(self._main_output_folder, 'featurewiz')
        output_file = os.path.join(output_folder, 'submission_file.csv')
        create_clean_folder(output_folder)
        config_override = {'output_folder': output_folder,
                          'train_set': self._train_set,
                          'test_set': self._test_set,
                          'output_file': output_file}
        main(config_path, config_override)

        df_submission = pd.read_csv(output_file)
        self.assertEqual(len(df_submission),39098)

    def test_nn_optuna_config(self):
        """
        Test execution with a valid configuration file.
        """
        config_path = os.path.join(self._test_folder, 'config_custom_transformer_nn_optuna.yaml')
        output_folder = os.path.join(self._main_output_folder, 'optuna')
        output_file = os.path.join(output_folder, 'submission_file.csv')
        create_clean_folder(output_folder)
        config_override = {'output_folder': output_folder,
                          'train_set': self._train_set,
                          'test_set': self._test_set,
                          'output_file': output_file}
        main(config_path, config_override)

        df_submission = pd.read_csv(output_file)
        self.assertEqual(len(df_submission),39098)

if __name__ == "__main__":
    unittest.main()
