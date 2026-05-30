import logging
from sklearn.kernel_ridge import KernelRidge
from misc import (
    load_data,
    preprocess_data,
    train_model,
    evaluate_model,
    display_results,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)


def main():
    logger.info("Starting KernelRidge pipeline")

    df = load_data()

    X_train, X_test, y_train, y_test = preprocess_data(df)

    model = KernelRidge(
        kernel='rbf',
        alpha=1.0
    )

    model = train_model(model, X_train, y_train)

    mse = evaluate_model(model, X_test, y_test)

    display_results("KernelRidge", mse)

    logger.info("Training pipeline completed successfully")


if __name__ == "__main__":
    main()