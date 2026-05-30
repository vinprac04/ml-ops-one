
"""
misc.py - Shared utility functions for ML pipeline.
"""

import logging
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error


logger = logging.getLogger(__name__)


def load_data():
    """
    Load the Boston Housing dataset from source URL
    and return as pandas DataFrame.
    """
    logger.info("Loading dataset")

    try:
        data_url = "http://lib.stat.cmu.edu/datasets/boston"
        raw_df = pd.read_csv(
            data_url,
            sep=r"\s+",
            skiprows=22,
            header=None
        )

        data = np.hstack([
            raw_df.values[::2, :],
            raw_df.values[1::2, :2]
        ])

        target = raw_df.values[1::2, 2]

        feature_names = [
            "CRIM",      # Per capita crime rate by town
            "ZN",        # Proportion of residential land zoned for large lots
            "INDUS",     # Proportion of non-retail business acres per town
            "CHAS",      # Charles River dummy variable (1 if tract bounds river, else 0)
            "NOX",       # Nitric oxide concentration (air pollution level)
            "RM",        # Average number of rooms per dwelling
            "AGE",       # Proportion of owner-occupied units built before 1940
            "DIS",       # Weighted distance to employment centres in Boston
            "RAD",       # Accessibility to radial highways
            "TAX",       # Property tax rate per $10,000
            "PTRATIO",   # Pupil–teacher ratio by town
            "B",         # Measure related to racial demographics in original dataset
            "LSTAT"      # Percentage of lower status population
        ]

        df = pd.DataFrame(data, columns=feature_names)
        df["MEDV"] = target

        logger.info("Dataset loaded successfully with shape %s", df.shape)

        return df

    except Exception as e:
        logger.exception("Error loading dataset: %s", e)
        raise


def preprocess_data(
    df,
    target_column="MEDV",
    test_size=0.2,
    random_state=42
):
    """
    Split dataset into train and test sets.
    """

    if target_column not in df.columns:
        raise ValueError(
            f"Target column '{target_column}' not found in dataframe"
        )

    X = df.drop(columns=[target_column])
    y = df[target_column]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        shuffle=True
    )

    logger.info(
        "Train/test split complete | train=%s test=%s",
        X_train.shape,
        X_test.shape
    )

    return X_train, X_test, y_train, y_test


def train_model(model, X_train, y_train):
    """
    Train sklearn-compatible model.
    """

    logger.info(
        "Training %s",
        type(model).__name__
    )

    model.fit(X_train, y_train)

    logger.info("Training completed")

    return model


def evaluate_model(model, X_test, y_test):
    """
    Evaluate model using Mean Squared Error.
    """

    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)

    logger.info(
        "Evaluation complete | MSE=%.4f RMSE=%.4f",
        mse,
        rmse
    )

    return mse


def display_results(model_name, mse):
    """
    Print final model results.
    """

    print("\n" + "=" * 50)
    print(f"Model: {model_name}")
    print(f"Test MSE: {mse:.4f}")
    print("=" * 50)

