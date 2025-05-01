from functools import partial
import os
import pickle
import numpy as np
from hyperopt import STATUS_OK, Trials, fmin, hp, tpe
from hyperopt.pyll import scope
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_validate
import pandas as pd
from sklearn.preprocessing import FunctionTransformer, LabelEncoder
from sklearn.linear_model import LogisticRegression
import pickle
import os

SOURCE = os.path.join("data", "processed")
MODEL_PATH = "models"
N_FOLDS = 3
MAX_EVALS = 3

# Updated search space with compatible parameters
SPACE = {
    "penalty": hp.choice("penalty", ["l1", "l2", "elasticnet"]),
    "C": hp.loguniform("C", -4, 4),
    "solver": hp.choice("solver", ["saga"]),  # Saga supports all penalties
    "l1_ratio": hp.uniform("l1_ratio", 0, 1),  # Required for elasticnet
}


def encode_target(file_name: str, target_col: str, model_name: str, logger):
    df_train = pd.read_parquet(os.path.join(SOURCE, f"{file_name}-train.parquet"))
    df_test = pd.read_parquet(os.path.join(SOURCE, f"{file_name}-test.parquet"))
    X_train, y_train = df_train.drop(columns=[target_col], axis=1), df_train[target_col]
    X_test, y_test = df_test.drop(columns=[target_col], axis=1), df_test[target_col]

    logger.info("Fitting the encoder/decoder of target variable")
    logger.info(f"Number of classes: {len(y_train.unique())}")
    """Create and fit encoder/decoder for target variable"""
    encoder = LabelEncoder()
    encoder.fit(y_train)
    # Create decoder mapping
    classes = encoder.classes_
    decoder = {i: cls for i, cls in enumerate(classes)}
    target_translator = {
        "encoder": encoder,
        "decoder": decoder,
    }
    logger.info("encoder/decoder of target created successfully")
    # Save the artifacts

    if not os.path.exists(os.path.join(MODEL_PATH, model_name)):
        os.makedirs(os.path.join(MODEL_PATH, model_name))
    with open(
        os.path.join(MODEL_PATH, model_name, "model_target_translator.pkl"),
        "wb",
    ) as pkl:
        pickle.dump(target_translator, pkl)
    logger.info("encoder/decoder of target saved")
    return X_train, y_train, X_test, y_test


def objective(params, X, y, n_folds: int = N_FOLDS):
    """Wrapper function for hyperparameter optimization"""
    try:
        # Handle elasticnet specific parameters
        if params["penalty"] == "elasticnet":
            params["l1_ratio"] = params.get("l1_ratio", 0.5)
        else:
            params.pop("l1_ratio", None)

        model = LogisticRegression(**params, max_iter=1000)
        scores = cross_validate(
            model,
            X,
            y,
            cv=n_folds,
            scoring="accuracy",
            error_score="raise",  # Get detailed errors
        )
        return {
            "loss": -np.mean(scores["test_score"]),  # Minimize negative accuracy
            "params": params,
            "status": STATUS_OK,
        }
    except Exception as e:
        return {"loss": 0, "status": STATUS_FAIL, "exception": str(e)}


def train_model(X, y, model_name: str, logger):
    """Complete training pipeline with error handling"""
    logger.info("Loading target encoder/decoder")
    try:
        with open(
            os.path.join(MODEL_PATH, model_name, "model_target_translator.pkl"), "rb"
        ) as pkl:
            translator = pickle.load(pkl)

        y_train_enc = translator["encoder"].transform(y)

        logger.info("Starting hyperparameter optimization")
        bayes_trials = Trials()

        best = fmin(
            fn=partial(objective, X=X, y=y_train_enc),
            space=SPACE,
            algo=tpe.suggest,
            max_evals=MAX_EVALS,
            trials=bayes_trials,
            show_progressbar=False,
        )

        # Get best parameters from trials
        best_params = bayes_trials.best_trial["result"]["params"]
        logger.info(f"Best parameters: {best_params}")

        # Train final model
        final_model = LogisticRegression(**best_params, max_iter=1000)
        final_model.fit(X, y_train_enc)

        # Save artifacts
        os.makedirs(os.path.join(MODEL_PATH, model_name), exist_ok=True)
        with open(os.path.join(MODEL_PATH, model_name, "final_model.pkl"), "wb") as pkl:
            pickle.dump(final_model, pkl)

        logger.info("Model trained and saved successfully")

    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        raise
