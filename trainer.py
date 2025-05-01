from src.logger import ExecutorLogger
from src.training.process_data import read_process_data
from src.training.evaluate import evaluate
from src.training.train import train_model, encode_target


def train_pipeline(logger: ExecutorLogger):
    # process pipeline
    numeric_features = ["Age", "Fare", "SibSp", "Parch"]
    categorical_features = ["Pclass", "Sex", "Embarked"]
    drop_features = ["PassengerId", "Name", "Ticket", "Cabin", "Survived"]
    logger.info("Training Started...")
    read_process_data(
        file_name="titanic",
        target="Survived",
        num_cols=numeric_features,
        cat_cols=categorical_features,
        drop_cols=drop_features,
        logger=logger,
    )
    X_train, y_train, X_test, y_test = encode_target(
        file_name="titanic",
        target_col="Survived",
        model_name="basemodel",
        logger=logger,
    )
    train_model(X=X_train, y=y_train, model_name="basemodel", logger=logger)
    evaluate(X_test, y_test, "basemodel", logger)
    logger.info("Training Completed...")


if __name__ == "__main__":
    logger = ExecutorLogger("train pipeline")
    train_pipeline(logger)
