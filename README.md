# Prediction of Quality of Wine

This project predicts wine quality using the **Wine Quality** dataset from Kaggle ([link]([https://www.kaggle.com/datasets/mehmetsabrikunt/internet-service-churn/data]). It involves data loading, cleaning, feature analysis, engineering, and addressing class imbalance with SMOTE. Machine learning models are evaluated and optimized to find the best predictor of wine quality based on chemical properties, with a structured workflow for each step.

## Table of Contents
1. [Project Overview](#project-overview)
2. [Dataset](#dataset)
3. [Data Loading](#data-loading)
4. [Feature Analysis](#feature-analysis)
5. [Data Cleaning](#data-cleaning)
6. [Imbalanced Handling (SMOTE)](#imbalanced-handling-smote)
7. [Feature Engineering](#feature-engineering)
8. [Model Analysis](#model-analysis)
9. [Model Optimization](#model-optimization)
10. [Visualizations](#visualizations)
11. [Future Work](#future-work)
12. [Contact](#contact)

## Project Overview

The aim of this project is to build a robust wine quality prediction model using machine learning techniques on the **Wine Quality** dataset. The process includes **Imbalanced Handling (SMOTE)** and detailed **Feature Engineering** steps such as **Extract New Features**, **Normalization**, and **Data type conversion**. 

Modeling techniques include:  
- **Hyperparameter Optimization Model**: RandomForest
- **Neural Network Models**: PyTorch, Keras  

Visualizations are presented in **Feature Analysis** and **Model Optimization**, focusing on data insights and performance metrics to identify the most accurate wine quality prediction model.

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

## Data Loading
In this section, we load and examine the raw data to understand its structure, variable types, and key statistics using the following scripts:

- **Unassign_the_Runtime_for_Resetting.py**: Clears the runtime environment to reset cached variables and prevent data conflicts.
- **Upload_the_data.py**: Loads the dataset and performs initial checks, including handling missing values and identifying data types.
- **Definitions_for_the_Columns.py**: Offers detailed descriptions and explanations of each column to improve data comprehension.

### Key Steps:
- Exploring the data: Inspecting missing values, data types, and class distribution.
- Generating summary statistics for both numerical and categorical features.

## Feature Analysis  
The Feature Analysis consists of five key parts: **Statistical Observation**, **Value Range**, **Numerical Features Relationship**, **Heatmap**, and **Class Imbalance**. These steps help in understanding the data distribution, relationships between features, and potential imbalances in the target variable. Further analysis will focus on gaining deeper insights to enhance model performance.


### Heatmap   
The heatmap below shows the correlation between different features and the `churn` variable, helping to identify the most relevant variables for analysis.

### Key Takeaways:

### Visualizations

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
