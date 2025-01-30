# **sklearn_pipelines_builder**
**Simplify, Optimize, and Benchmark Your Machine Learning Workflows**

---

## 🚀 **Overview**

**sklearn_pipelines_builder** is a framework under development, designed to streamline the creation and benchmarking of complex machine learning pipelines.

Machine learning is notoriously experimental, requiring practitioners to conduct numerous experiments to achieve optimal results. With sklearn_pipelines_builder, you can easily design, test, and benchmark different strategies, making it simple to identify the most effective methods for your data.

This package empowers data scientists to build modular, reusable pipelines with integrated support for tools like **MLFlow**, **AutoGluon**, **Optuna**, **Featurewiz**, and **FeatureEngine**, enabling seamless tracking, optimization, and performance comparison.

---

## ✨ **Key Features**

- **Modular Pipeline Design**: Quickly build pipelines using YAML or Python configurations.
- **Seamless Benchmarking**: Effortlessly compare multiple strategies or models in one framework.
- **Integrated AutoML Support**: Leverage AutoGluon, Optuna, and other tools for hyperparameter tuning and AutoML workflows.
- **Experiment Tracking**: Track metrics, configurations, and artifacts with MLFlow.
- **Rich Preprocessing Options**: Includes advanced feature engineering, selection, and imputation methods.
- **Extensibility**: Add your custom transformers, estimators, or cross-validation strategies.

---

## 🛠️ **Core Functionality**

### **1. YAML-Based Configuration**
Define your pipeline in YAML for a clear, reproducible workflow.

#### **Example Configuration**
```yaml
experiment_name: playground-series-s4e11
run_name: custom_transformer_feature_engine_rfe

validation:
  n_folds: 5
scoring: 'accuracy'
output_folder: D:\Kaggle\playground-series-s4e11\output
train_set: D:\Kaggle\playground-series-s4e11\train.csv
test_set: D:\Kaggle\playground-series-s4e11\test.csv
response: loan_status
prediction_type: class
output_file: custom_transformer_accuracy.csv
output_columns: id
pipe_line_steps:
  - element_type: rare_value_transform
  - element_type: drop_columns
    columns: id
  - element_type: label_encoder
  - element_type: multi_column_imputer
    actual_imputer_config:
      element_type: select_best_imputer
      imputation_methods:
        - element_type: pipe_line_aggregation
          element_alias: knn_regressor_4_features
          elements:
            - element_type: select_k_best
              score_func_name: decision_tree_regressor
              k: 4
            - element_type: knn_regressor
        - element_type: mean_predictor
        - element_type: most_frequent
  - element_type: feature_engine_rfe
    threshold: 0.000001
    model_config:
      element_type: catboost_classifier
final_step:
  element_type: optuna_model_optimizer
  model_name: catboost_classifier
  n_trials: 5
  storage_type: sqlite
```

### **2. AutoML Integration**

Easily integrate AutoML tools like AutoGluon and Optuna for:

Hyperparameter tuning
Feature selection
Automated model benchmarking
### **3. Feature Selection**
Supports advanced feature selection methods using:

Recursive Feature Elimination (RFE): Powered by FeatureEngine.
Featurewiz: Automatically selects important features based on mutual information, correlation, and predictive power.
### **4. Benchmarking with MLFlow**
Track experiments and results, making it simple to benchmark different strategies:

## **📝 Quickstart Guide**
### **Step 1: Define Your Pipeline**
Write a YAML file as shown above to configure your pipeline.

### **Step 2: Execute the Pipeline**
Run your pipeline with a single command:


python sklearn_pipelines_builder/main.py --config pipeline.yaml
### ** Step 3: Visualize Results**
Use MLFlow to compare results and benchmark your strategies:

mlflow ui
## **🔧 Installation**
Currently, sklearn_pipelines_builder is under development. Clone the repository to try it out:

git clone https://github.com/jonathanhexner/sklearn_pipelines_builder.git

cd sklearn_pipelines_builder

pip install -r requirements.txt
## **📚 Features at a Glance**
Feature	Description
YAML Pipelines	Modular, configurable workflows for machine learning.
Integrated AutoML	Support for AutoGluon, Optuna, and similar tools.
Recursive Feature Elimination	Advanced feature selection with cross-validation and custom thresholds.
Featurewiz Integration	Automatically identifies important features for predictive tasks.
Imputation	Smart handling of missing data with configurable strategies.
Benchmarking	Track metrics, configurations, and results using MLFlow.
## **🌟 Planned Features**
Template Pipelines: Prebuilt configurations for tasks like classification, regression, and time-series analysis.
Interactive Dashboards: Enhanced visualization for feature importance and benchmarking results.
## **🤝 Contributing**
We welcome contributions! See the CONTRIBUTING.md file for details.

## **📄 License**
Distributed under the MIT License. See LICENSE for more information.

## **Contact**

📧 Jonathan Hexner: jonathan.hexner@gmail.com

🌐 GitHub: github.com/jonathanhexner


