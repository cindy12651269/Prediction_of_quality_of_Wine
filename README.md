# Prediction of Quality of Wine

This project focuses on predicting wine quality using machine learning techniques. We utilize the **Wine Quality** dataset, which contains various chemical properties of wines. The objective is to explore and experiment with different models to see which one can best predict the quality of wines based on their characteristics.

## Table of Contents

- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Data Preprocessing](#data-preprocessing)
- [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
- [Feature Engineering](#feature-engineering)
- [Model Selection](#model-selection)
- [Results](#results)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)

## Project Overview

The aim of this project is to build a machine learning pipeline that predicts wine quality based on a variety of chemical features. We use several classification models and evaluate their performance on this task. The final goal is to identify which model performs best and provides the most accurate predictions of wine quality. We also emphasize data preprocessing, feature engineering, and exploratory data analysis (EDA) to gain insights into the data and improve model accuracy.

## Dataset

The **Wine Quality** dataset used in this project is sourced from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/wine+quality). It contains chemical measurements of red and white wines, such as:

- **Fixed Acidity**: Measures the concentration of non-volatile acids in wine.
- **Volatile Acidity**: Refers to acetic acid in wine, which at high levels can lead to an unpleasant vinegar taste.
- **Citric Acid**: A natural preservative that adds freshness and flavor.
- **Residual Sugar**: The amount of sugar left after fermentation, influencing sweetness.
- **Chlorides**: The amount of salt in the wine.
- **Free Sulfur Dioxide**: Prevents microbial growth and oxidation in wine.
- **Total Sulfur Dioxide**: The total level of SO2, which can affect the flavor if too high.
- **Density**: A factor that can help determine alcohol and sugar levels in wine.
- **pH**: A measure of the acidity/basicity of the wine.
- **Sulphates**: A wine preservative contributing to both the bitterness and microbial stability.
- **Alcohol**: Alcohol content affects the perceived quality and taste.

The **target variable** is the **quality** of wine, scored between 0 and 10, where higher scores represent better quality.

## Data Preprocessing

Data preprocessing is a critical step in machine learning pipelines. In this project, we handle missing values, outliers, and normalization of features to prepare the data for modeling. The key steps include:

- **Handling Missing Data**: If any features contain missing values, we explore methods such as mean imputation or removing incomplete records.
- **Normalization**: Given that the features have varying scales (e.g., alcohol percentage vs. pH), we standardize or normalize the data to ensure better performance for models sensitive to feature scaling, such as logistic regression or SVM.
- **Outlier Detection**: Extreme values in features like volatile acidity or residual sugar could skew the model results, so we detect and, if necessary, remove outliers.

## Exploratory Data Analysis (EDA)

Before building any model, we conduct EDA to understand the distribution of data, relationships between features, and correlation with the target variable. Some key steps in our EDA include:

- **Histograms and Boxplots**: We visualize the distribution of each feature to understand their range, skewness, and identify potential outliers.
- **Correlation Matrix**: This helps us understand the relationship between features and how they correlate with the target variable (wine quality).
- **Pair Plots**: These allow us to explore relationships between pairs of features and their impact on wine quality.
- **Feature Importance**: Using techniques such as permutation importance, we analyze which features contribute most to predicting wine quality.

## Feature Engineering

Feature engineering involves creating new features or modifying existing ones to improve the model's performance. In this project, we explore several feature engineering techniques, such as:

- **Polynomial Features**: We create polynomial combinations of existing features (e.g., `pH^2`, `Alcohol * Residual Sugar`) to capture non-linear relationships.
- **Interaction Terms**: Interaction terms are generated between features that may have a combined effect on the target variable.
- **Binning**: We create bins for continuous variables like alcohol content and residual sugar to reduce noise and improve model interpretability.

## Model Selection

We experiment with a variety of classification models to predict wine quality. These models include:

- **Logistic Regression**: A linear model used for binary classification, applied here to predict multi-class labels by converting the quality score into a classification problem.
- **Decision Tree**: A model that splits the data into branches to predict the quality of wine. It is simple and interpretable but prone to overfitting.
- **Random Forest**: An ensemble model that builds multiple decision trees and averages their predictions, often yielding better results than a single decision tree.
- **Support Vector Machines (SVM)**: A model that finds the optimal hyperplane that separates different classes. SVM is particularly useful for complex, non-linear boundaries.
- **Gradient Boosting**: A boosting algorithm that builds models sequentially, each new model correcting errors made by the previous one. This often leads to high accuracy.

We use **GridSearchCV** to fine-tune hyperparameters for each model to achieve optimal performance.

## Results

The models are evaluated using various performance metrics, including:

- **Accuracy**: The percentage of correctly classified instances.
- **Precision**: The proportion of predicted positive cases that are actually positive.
- **Recall**: The proportion of actual positive cases that were predicted correctly.
- **F1 Score**: The harmonic mean of precision and recall, providing a balanced metric for classification.
- **Confusion Matrix**: This is used to visualize the true positives, true negatives, false positives, and false negatives for the predictions.

The final model achieves an accuracy of approximately **X%** on the test set (fill in with actual results after training and testing).

## Usage

1. **Data Loading**: The dataset is loaded and split into training and validation sets.
2. **Model Training**: Multiple models are trained on the training set, and their performance is evaluated.
3. **Model Evaluation**: The model with the best performance is selected based on accuracy, precision, recall, and F1-score.
4. **Predictions**: The final model is used to predict the quality of wine on new unseen data.

To run the project locally, follow these steps:

- Open the Jupyter notebook file `Prediction_of_quality_of_Wine.ipynb`.
- Run all the cells to load data, preprocess it, train models, and evaluate their performance.

## Contributing

If you want to contribute to this project, feel free to fork the repository and submit a pull request. Improvements in model performance or suggestions for new features are always welcome.

## License

This project is licensed under the MIT License. See the LICENSE file for more information.
