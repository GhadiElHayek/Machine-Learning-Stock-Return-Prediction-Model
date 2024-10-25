# Machine-Learning-Stock-Return-Prediction-Model
A machine learning-based model for predicting JNJ stock returns using logistic regression, LDA, QDA, LASSO, and Ridge regression. The model processes multiple financial datasets and evaluates performance using confusion matrices, ROC curves, and AUC.
Stock Return Prediction Model

Overview

This project employs machine learning algorithms to predict binary outcomes of JNJ stock returns using various financial indicators from GS, MCD, MMM, and PG stock data. It uses multiple classification techniques, including logistic regression, LDA, QDA, LASSO, and Ridge regression, and evaluates model performance with confusion matrices, ROC curves, and AUC.

Features

Data Processing
Financial Data: Processes return data for JNJ, GS, MCD, MMM, and PG stocks.
Lagged Variables: Creates lagged returns for each independent variable to enhance the prediction capabilities of the model.
Data Merging: Combines multiple datasets on the exchange date for a unified dataset.
Machine Learning Models
Logistic Regression: Baseline model for predicting the probability of JNJ stock returns.
LDA (Linear Discriminant Analysis): Uses linear boundaries to classify binary outcomes.
QDA (Quadratic Discriminant Analysis): Uses non-linear boundaries for classification.
LASSO Regression: Applies L1 regularization to the logistic regression model.
Ridge Regression: Applies L2 regularization to the logistic regression model.
Model Evaluation
Confusion Matrices: Displays true positives, false positives, true negatives, and false negatives for each model.
ROC Curves: Plots ROC curves to visualize the trade-off between the true positive rate and false positive rate.
AUC (Area Under Curve): Measures the model’s ability to classify binary outcomes correctly.
Technologies Used

R: Core programming language.
caret: For machine learning model training and data partitioning.
ROCR & pROC: For ROC curve plotting and AUC calculation.
glmnet: For LASSO and Ridge regression.
MASS: For LDA and QDA analysis.
dplyr: For data manipulation.
