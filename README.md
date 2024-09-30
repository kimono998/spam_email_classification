# Email Spam Classifier

## Problem Statement

The goal of this project is to build a classifier that can effectively distinguish between spam and non-spam emails while minimizing the number of false positives. The project was done as a part of my ML-IDA course. 

## Preprocessing

The dataset consists of emails represented by sparse bag-of-words feature vectors. The initial feature space contains 57,173 attributes, but the dictionary mapping attribute positions to words is not provided.

To reduce dimensionality, words that appear fewer than 3 times across all documents were removed. This results in a feature space of approximately 49,171 attributes from an original set of 57,173.

- **Initial Feature Space**: 57,173 attributes
- **Reduced Feature Space**: 10,000 samples × 49,171 attributes

## Model Selection

The following classifiers were trained and evaluated using features from `scikit-learn`. Hyperparameters were optimized using `GridSearchCV`, and the models were evaluated using a holdout validation split of 80:20. The primary evaluation metrics were `roc_auc` and `f1`.

### Models

1. **Support Vector Machine (SVM)**
    - Optimizer: Stochastic Gradient Descent (SGD)
    - Loss Function: Hinge Loss
    - Regularization: L2
    - Lambda: 1e-05

2. **Logistic Regression**
    - Optimizer: Stochastic Gradient Descent (SGD)
    - Loss Function: Logistic Loss
    - Regularization: L2
    - Lambda: 1e-05

3. **Random Forest Classifier**
    - `n_estimators`: 1,000
    - `min_samples_split`: 10
    - `min_samples_leaf`: 1
    - `max_features`: 'auto'
    - `max_depth`: 80
    - `bootstrap`: False

4. **Ensemble Voting Classifier**
    - Voting: 'hard'

## Evaluation Metrics

The primary metrics used to evaluate the models were:
- **ROC AUC**
- **F1 Score**
