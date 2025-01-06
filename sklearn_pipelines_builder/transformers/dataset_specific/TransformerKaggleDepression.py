import numpy as np
import pandas as pd
from sklearn_pipelines_builder.infrastructure.BaseConfigurableTransformer import BaseConfigurableTransformer
from sklearn_pipelines_builder.SingletonContainer import SingleContainer
from sklearn_pipelines_builder.utils.basic_utils import remove_from_list
from sklearn_pipelines_builder.utils.collect_info import collect_info

class TransformerKaggleDepression(BaseConfigurableTransformer):
    def __init__(self, config={}):
        self.value_counts_ = {}
        self.config = config

    def fit(self, X, y=None):
        """
        Compute the value counts for categorical columns with rare values.
        """
        # Store value counts for specific columns to identify rare values
        for col in ['City', 'Profession', 'Degree']:
            self.value_counts_[col] = X[col].value_counts(normalize=True)
        return self

    def transform(self, X_in):
        """
        Apply the transformations to the dataset.
        """
        # Copy the input DataFrame to avoid modifying the original data
        X = X_in.copy()

        # Gender Mapping
        X['Gender'] = X['Gender'].map({'Male': 0, 'Female': 1})
        SingleContainer.numeric_features += ['Gender']
        SingleContainer.string_features = remove_from_list(SingleContainer.string_features,
                                                            ['Gender'])
        # Replace CGPA for working professionals
        X.loc[X['Working Professional or Student'] == 'Working Professional', 'CGPA'] = -1

        # Handle missing 'Profession' for students
        X.loc[(X['Working Professional or Student'] == 'Student') & (X['Profession'].isnull()), 'Profession'] = 'Student'


        # Combine satisfaction columns into one
        string_columns_added = ['Satisfaction']
        string_columns_remove = ['Job Satisfaction', 'Study Satisfaction']
        X['Satisfaction'] = np.nan
        X.loc[X['Working Professional or Student'] == 'Working Professional', 'Satisfaction'] = \
            X.loc[X['Working Professional or Student'] == 'Working Professional', 'Job Satisfaction']
        X.loc[X['Working Professional or Student'] == 'Student', 'Satisfaction'] = \
            X.loc[X['Working Professional or Student'] == 'Student', 'Study Satisfaction']
        self.udpate_columns(X, string_columns_added, string_columns_remove)

        # Combine pressure columns into one
        string_columns_added = ['Pressure']
        string_columns_remove = ['Work Pressure', 'Academic Pressure']
        X['Pressure'] = np.nan
        X.loc[X['Working Professional or Student'] == 'Working Professional', 'Pressure'] = \
            X.loc[X['Working Professional or Student'] == 'Working Professional', 'Work Pressure']
        X.loc[X['Working Professional or Student'] == 'Student', 'Pressure'] = \
            X.loc[X['Working Professional or Student'] == 'Student', 'Academic Pressure']
        self.udpate_columns(X, string_columns_added, string_columns_remove)


        # Map suicidal thoughts to binary
        X['Have you ever had suicidal thoughts ?'] = \
            X['Have you ever had suicidal thoughts ?'].map({'No': 0, 'Yes': 1})
        SingleContainer.numeric_features += ['Have you ever had suicidal thoughts ?']
        SingleContainer.string_features = remove_from_list(SingleContainer.string_features,
                                                            ['Have you ever had suicidal thoughts ?'])

        # Map sleep duration to numerical categories
        X['Sleep_Duration'] = X['Sleep Duration'].map({
            'Less than 5 hours': 0,
            '5-6 hours': 1,
            '7-8 hours': 2,
            'More than 8 hours': 3
        })
        X.drop(columns=['Sleep Duration'], inplace=True)
        SingleContainer.numeric_features += ['Sleep_Duration']
        SingleContainer.string_features = remove_from_list(SingleContainer.string_features,
                                                            ['Sleep Duration'])
        # Map dietary habits to numerical categories
        X['Dietary_Habits'] = X['Dietary Habits'].map({
            'Unhealthy': 0,
            'Moderate': 1,
            'Healthy': 2
        })
        X.drop(columns=['Dietary Habits'], inplace=True)
        SingleContainer.numeric_features += ['Dietary Habits']
        SingleContainer.string_features = remove_from_list(SingleContainer.string_features,
                                                            ['Dietary Habits'])

        # Replace rare values with NaN for specified columns
        for col in ['City', 'Profession', 'Degree']:
            invalid_values = self.value_counts_[col][self.value_counts_[col] < 0.01].index.tolist()
            X.loc[X[col].isin(invalid_values), col] = np.nan

        SingleContainer.all_features = list(set(X.columns)-set([SingleContainer.response]))
        collect_info(X)
        return X

    def udpate_columns(self, X, string_columns_added, string_columns_remove):
        X.drop(columns=string_columns_remove, inplace=True)
        SingleContainer.columns_to_drop += string_columns_remove
        SingleContainer.string_features += string_columns_added
        SingleContainer.string_features = remove_from_list(SingleContainer.string_features, string_columns_remove)
