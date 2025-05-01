import os
import pickle
from functools import partial

import numpy as np
import pandas as pd
from hyperopt import STATUS_OK, STATUS_FAIL, Trials, fmin, hp, tpe
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_validate
from sklearn.preprocessing import LabelEncoder


def encode_target(model_cfg, logger):
    processed_path = model_cfg.processed_data_path
    file_name = model_cfg.file_name
    target_col = model_cfg.target
    model_path = os.path.join(model_cfg.model_path, model_cfg.model_name)

    df_train = pd.read_parquet(os.path.join(processed_path, f"{file_name}-train.parquet"))
    df_test = pd.read_parquet(os.path.join(processed_path, f"{file_name}-test.parquet"))
    X_train, y_train = df_train.drop(columns=[target_col]), df_train[target_col]
    X_test, y_test = df_test.drop(columns=[target_col]), df_test[target_col]

    logger.info("Fitting the encoder/decoder of target variable")
    encoder = LabelEncoder()
    encoder.fit(y_train)
    decoder = {i: cls for i, cls in enumerate(encoder.classes_)}
    target_translator = {"encoder": encoder, "decoder": decoder}

    os.makedirs(model_path, exist_ok=True)
    with open(os.path.join(model_path, "model_target_translator.pkl"), "wb") as f:
        pickle.dump(target_translator, f)

    logger.info("encoder/decoder of target saved")
    return X_train, y_train, X_test, y_test


def get_search_space(space_cfg):
    def convert(val):
        if isinstance(val, str) and val.startswith("choice("):
            options = val[len("choice("):-1].split(",")
            return hp.choice(str(hash(val)), [o.strip() for o in options])
        elif isinstance(val, str) and val.startswith("loguniform("):
            low, high = map(float, val[len("loguniform("):-1].split(","))
            return hp.loguniform(str(hash(val)), low, high)
        elif isinstance(val, str) and val.startswith("uniform("):
            low, high = map(float, val[len("uniform("):-1].split(","))
            return hp.uniform(str(hash(val)), low, high)
        return val  # as-is
    return {k: convert(v) for k, v in space_cfg.items()}


def objective(params, X, y, n_folds):
    try:
        if params.get("penalty") != "elasticnet":
            params.pop("l1_ratio", None)

        model = LogisticRegression(**params, max_iter=1000)
        scores = cross_validate(model, X, y, cv=n_folds, scoring="accuracy", error_score="raise")
        return {
            "loss": -np.mean(scores["test_score"]),
            "params": params,
            "status": STATUS_OK,
        }
    except Exception as e:
        return {"loss": 0, "status": STATUS_FAIL, "exception": str(e)}


def train_model(X, y, cfg, logger):
    model_cfg = cfg
    model_path = os.path.join(model_cfg.model_path, model_cfg.model_name)

    logger.info("Loading target encoder/decoder")
    with open(os.path.join(model_path, "model_target_translator.pkl"), "rb") as f:
        translator = pickle.load(f)

    y_enc = translator["encoder"].transform(y)

    logger.info("Starting hyperparameter optimization")
    trials = Trials()
    search_space = get_search_space(model_cfg.hyperparameters.space)

    best = fmin(
        fn=partial(objective, X=X, y=y_enc, n_folds=model_cfg.hyperparameters.n_folds),
        space=search_space,
        algo=tpe.suggest,
        max_evals=model_cfg.training.max_evals,
        trials=trials,
        show_progressbar=False,
    )

    best_params = trials.best_trial["result"]["params"]
    logger.info(f"Best parameters: {best_params}")

    final_model = LogisticRegression(**best_params, max_iter=model_cfg.training.max_iter)
    final_model.fit(X, y_enc)

    with open(os.path.join(model_path, "final_model.pkl"), "wb") as f:
        pickle.dump(final_model, f)

    logger.info("Model trained and saved successfully")
