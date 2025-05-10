import os
import pickle
from functools import partial
from dotenv import load_dotenv
import numpy as np
import pandas as pd
from hyperopt import STATUS_OK, STATUS_FAIL, Trials, fmin, hp, tpe
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_validate
from sklearn.preprocessing import LabelEncoder
from omegaconf import OmegaConf
import dvc.api
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.logger import ExecutorLogger
from types import SimpleNamespace
import dagshub
import mlflow
# Add this before loading the pipeline
from src.training.process_data import dtype_conversion

# Add explicit registration for pickle
import sys
def dtype_conversion(X, cat_cols):
    X = X.copy()
    for col in cat_cols:
        if col in X.columns:
            # Fill NA and convert to category
            X[col] = X[col].fillna('missing').astype('category')
    return X
from src.training.model_wrapper import ModelWrapper

def dict_to_namespace(d):
    for k, v in d.items():
        if isinstance(v, dict):
            d[k] = dict_to_namespace(v)
    return SimpleNamespace(**d)

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

    return {k: convert(v) for k, v in vars(space_cfg).items()}

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

    params = trials.best_trial["result"]["params"]
    logger.info(f"Best parameters: {params}")



    
    # New:
    with mlflow.start_run():
        mlflow.autolog()
        final_model = LogisticRegression(**params, max_iter=model_cfg.training.max_iter)
        final_model.fit(X, y_enc)
        logger.info("save the final optimized model")
        if not os.path.exists(
            os.path.join(model_cfg.model_path, model_cfg.model_name)
        ):
            os.makedirs(
                os.path.join(
                    model_cfg.model_path, 
                    model_cfg.model_name
                )
            )
        with open(
            os.path.join(
                model_cfg.model_path, 
                model_cfg.model_name, 
                "final_model.pkl"
            ),
            "wb",
        ) as pkl:
            pickle.dump(final_model, pkl)
        logger.info("model trained and saved successfully")
        run_id = mlflow.active_run().info.run_id
        train_preds = final_model.predict(X)
        signature = mlflow.models.infer_signature(X, train_preds)
        mlflow.pyfunc.log_model(
            model_cfg.model_name, 
            python_model=ModelWrapper(),
            artifacts={ 
                'encoder': os.path.join(
                    model_cfg.model_path,
                    model_cfg.model_name,
                    "model_target_translator.pkl",
                ),
                'model': os.path.join(
                    model_cfg.model_path, 
                    model_cfg.model_name, 
                    "final_model.pkl"
                )
            },
            signature=signature,
            registered_model_name=model_cfg.model_name,
        )
        mlflow.log_params(params)
        mlflow.log_metrics({
            f"cv_Accuracy_score": trials.best_trial["result"]["loss"]
        })
        artifact_path = "model"
        model_uri = f"runs:/{run_id}/{artifact_path}"

        model_details = mlflow.register_model(
            model_uri=model_uri, 
            name=model_cfg.model_name
        )
        logger.info("Model registered successfully!!")

        return model_details, run_id

    
    
    
    
    

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

def move_model_to_prod(client, model_details, logger) -> None:
    client.transition_model_version_stage(
        name=model_details.name,
        version=model_details.version,
        stage="production",
    )
    logger.info("Model transitioned to prod stage")




def setup_mlflow(tracking_uri: str, logger):
    mlflow.set_tracking_uri(tracking_uri)
    client = mlflow.client.MlflowClient(tracking_uri=tracking_uri)
    logger.info("MLFlow Client Defined and tracking URI Setted Successfully.")
    return client

if __name__ == "__main__":

    logger = ExecutorLogger("dvc-training")
    load_dotenv(".env")
    cfg = dvc.api.params_show()
    cfg_model = dict_to_namespace(cfg["model"])

    dagshub.auth.add_app_token(token=os.getenv("DAGSHUB_TOKEN"))
    dagshub.init(
        repo_owner=os.getenv("DAGSHUB_USERNAME"), 
        repo_name=cfg["model"]["repo_name"], 
        mlflow=cfg["model"]["use_mlflow"]
    )
    client = setup_mlflow(cfg["model"]["tracking_uri"], logger)


    X_train, y_train, X_test, y_test = encode_target(cfg_model, logger)
    model_details, run_id = train_model(X_train, y_train, cfg_model, logger)
    move_model_to_prod(client, model_details, logger)