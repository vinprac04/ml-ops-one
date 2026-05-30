"""
misc.py - Shared utility functions for ML pipeline.

"""

import os
import logging
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

## Load data from the Boston housing Data set for the provided data as per the assignment files
def load_data():
    data_url = "http://lib.stat.cmu.edu/datasets/boston"
    raw_df = pd.read_csv(data_url, sep=r"\s+", skiprows=22, header=None)
    
    # data split
    data = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])
    target = raw_df.values[1::2, 2]

    feature_names = [
        'CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE',
        'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT'
    ]

    df = pd.DataFrame(data, columns=feature_names)
    df['MEDV'] = target
    return df


def preprocess_data(df, target_column='MEDV', test_size=0.2, random_state=42):
    
    ###Split data into features (X) and target (y), then into train/test sets.
    logger.info("preprocess_data() called with target_column='%s', test_size=%.2f, random_state=%d",
                target_column, test_size, random_state)

    X = df.drop(columns=[target_column])
    y = df[target_column]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    return X_train, X_test, y_train, y_test


def train_model(model, X_train, y_train):
    """
    Train any model which is passed as paramter
    """
    logger.info("train_model() called with model: %s", type(model).__name__)
    model.fit(X_train, y_train)
    logger.info("Model training complete: %s is now fitted", type(model).__name__)
    return model


def test_model(model, X_test, y_test):
    """
    Evaluate a trained model on test data using Mean Squared Error (MSE).
    """
    logger.info("evaluate_model() called with model: %s", type(model).__name__)
    y_pred = model.predict(X_test)
    logger.info("Predictions generated, computing Mean Squared Error...")
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    logger.info("MSE: %.4f | RMSE: %.4f ",mse, rmse)
   # logger.info("Prediction range: [%.2f, %.2f] vs Actual range: [%.2f, %.2f]",
              #  y_pred.min(), y_pred.max(), y_test.min(), y_test.max())
    return mse


def evaluate_model_cv(model, df, n_splits=5):
    """
    Evaluate a model by running multiple random train/test splits and averaging the MSE.

    Calls preprocess_data(), train_model(), and test_model() in each iteration.

        avg_mse: mean MSE across all splits
        mse_scores: list of per-split MSE values
    """
    from sklearn.base import clone
    logger.info("evaluate_model_cv() called with model: %s, n_splits=%d", type(model).__name__, n_splits)
    mse_scores = []
    for i in range(n_splits):
        X_train, X_test, y_train, y_test = preprocess_data(df, random_state=i)
        fold_model = clone(model)
        fold_model = train_model(fold_model, X_train, y_train)
        mse = test_model(fold_model, X_test, y_test)
        mse_scores.append(mse)
    avg_mse = np.mean(mse_scores)
    logger.info("Per-split MSE scores: %s", mse_scores)
    logger.info("Average MSE: %.4f", avg_mse)
    return avg_mse, mse_scores


def display_results(model_name, mse):
    """
    Print the evaluation results in a clean format.

    PARAMETERS:
        model_name: string name of the model (for display)
        mse: the computed MSE score
    """
    logger.info("display_results() called for model: %s", model_name)
    print(f"{'='*50}")
    print(f"Model: {model_name}")
    print(f"Average MSE on test set: {mse:.4f}")
    print(f"{'='*50}")
    logger.info("Results displayed for %s — MSE: %.4f", model_name, mse)
