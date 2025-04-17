import os

output_folder = os.path.join(r'D:\Kaggle\rohlik-sales-forecasting-challenge-v2', 'try1')
config_path = os.path.join(r'D:\Kaggle\rohlik-sales-forecasting-challenge-v2\rohlik_sales_try.yaml', 'config_custom_transformer_catboost_optuna.yaml')

output_file = os.path.join(output_folder, 'submission_file.csv')
create_clean_folder(output_folder)
config_override = {'output_folder': output_folder,
                   'train_set': self._train_set,
                   'test_set': self._test_set,
                   'output_file': output_file}
main(config_path, config_override)
