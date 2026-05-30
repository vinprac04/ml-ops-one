"""
train2.py - Train a Kernel Ridge Regression model on the Boston Housing dataset.

"""

import logging

from sklearn.kernel_ridge import KernelRidge
from misc import load_data, evaluate_model_cv, display_results

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


def main():
    logger.info("="*60)
    logger.info("Starting train2.py — KernelRidge pipeline")
    logger.info("="*60)

    logger.info("Step 1: Loading Boston Housing dataset...")
    df = load_data()

    logger.info("Step 2: Preparing features and target...")
    X = df.drop(columns=['MEDV'])
    y = df['MEDV']

    logger.info("Step 3: Evaluating KernelRidge with 5-fold cross-validation...")
    model = KernelRidge(kernel='rbf', alpha=1.0)
    avg_mse, fold_scores = evaluate_model_cv(model, df, n_splits=5)
    display_results("KernelRidge", avg_mse)

    logger.info("Pipeline complete for KernelRidge")


if __name__ == "__main__":
    main()
