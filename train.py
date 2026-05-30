import logging
from sklearn.tree import DecisionTreeRegressor
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
    logger.info("Starting DecisionTreeRegressor pipeline")

    df = load_data()

    X_train, X_test, y_train, y_test = preprocess_data(df)

    model = DecisionTreeRegressor(
        max_depth=5,
        min_samples_split=10,
        min_samples_leaf=4,
        random_state=42
    )

    model = train_model(model, X_train, y_train)

    mse = evaluate_model(model, X_test, y_test)

    display_results("DecisionTreeRegressor", mse)

    logger.info("Training pipeline completed successfully")


if __name__ == "__main__":
    main()