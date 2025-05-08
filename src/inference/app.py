# import os
# import pickle
# from functools import partial
# from dotenv import load_dotenv
# import sys
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
# import numpy as np
# import pandas as pd
# import mlflow
# import pandas as pd
# import dvc.api
# from hyperopt import STATUS_OK, STATUS_FAIL, Trials, fmin, hp, tpe
# from sklearn.linear_model import LogisticRegression
# from sklearn.model_selection import cross_validate
# from sklearn.preprocessing import LabelEncoder
# from omegaconf import OmegaConf

# from src.logger import ExecutorLogger
# from src.inference.requests import InferenceRequest
# from types import SimpleNamespace
# import dagshub
# import mlflow

# from src.training.model_wrapper import ModelWrapper


# import numpy as np
# import litserve as ls


# def dict_to_namespace(d):
#     for k, v in d.items():
#         if isinstance(v, dict):
#             d[k] = dict_to_namespace(v)
#     return SimpleNamespace(**d)

# # def setup_mlflow(tracking_uri: str):
# #     mlflow.set_tracking_uri(tracking_uri)
# #     client = mlflow.client.MlflowClient(tracking_uri=tracking_uri)
# #     return client
    
# # def get_model_runID(logged_model):
# #     load_dotenv(".env")
# #     cfg = dvc.api.params_show()
# #     cfg_model = dict_to_namespace(cfg["model"])

# #     cfg = cfg_model
# #     model_name = cfg.model_name
# #     MODEL_PATH = cfg.model_path
# #     PROCESSED_PATH = cfg.processed_data_path
# #     FILE_NAME = cfg.file_name
# #     TARGET = cfg.target


# #     # df_test = pd.read_parquet(os.path.join(PROCESSED_PATH, f"{FILE_NAME}-test.parquet"))
# #     # X_test, y_test = df_test.drop(columns=[TARGET]), df_test[TARGET]


# #     dagshub.auth.add_app_token(token=os.getenv("DAGSHUB_TOKEN"))
# #     dagshub.init(
# #         repo_owner=os.getenv("DAGSHUB_USERNAME"), 
# #         repo_name=cfg_model.repo_name, 
# #         mlflow=cfg_model.use_mlflow
# #     )
# #     client = setup_mlflow(cfg_model.tracking_uri)
# #     loaded_model = mlflow.pyfunc.load_model(logged_model)
# #     # data = X_test.copy()
# #     # loaded_model.predict(data)
# #     return loaded_model


# class InferenceAPI(ls.LitAPI):
#     # def setup(self, device = "cpu"):
#     #     logged_model = 'runs:/5dd549707b534d61b618b80c10e3868a/LogisticRegression'
#     #     self._model = get_model_runID(logged_model)
#     def setup(self, device="cpu"):
#         load_dotenv(".env")
#         cfg = dvc.api.params_show()
#         cfg_model = dict_to_namespace(cfg["model"])

#         dagshub.auth.add_app_token(os.getenv("DAGSHUB_TOKEN"))
#         dagshub.init(
#             repo_owner=os.getenv("DAGSHUB_USERNAME"),
#             repo_name=cfg_model.repo_name,
#             mlflow=cfg_model.use_mlflow,
#         )
#         mlflow.set_tracking_uri(cfg_model.tracking_uri)

#         # 1) Load the pyfunc model
#         logged_model = f"runs:/{self.run_id}/model"
#         self._model = mlflow.pyfunc.load_model(logged_model)

#         # 2) Manually load the encoder/decoder artifact
#         model_dir = os.path.join(cfg_model.model_path, cfg_model.model_name)
#         with open(os.path.join(model_dir, "model_target_translator.pkl"), "rb") as f:
#             translator = pickle.load(f)
#         self._decoder = translator["decoder"]
#     def decode_request(self, request):
#         # assume request["input"] is a dict of feature values
#         data = list(request["input"].values())
#         arr = np.array(data)[None, :]  # shape (1, n_features)
#         return arr

#     def predict(self, x):
#         if x is not None:
#             return self._model.predict(x)
#         else:
#             return None

#     def encode_response(self, output):
#         if output is None:
#             return {"message": "Error Occurred", "prediction": None}
#         # translate numeric codes back to original labels
#         decoded = [ self._decoder[int(v)] for v in output ]
#         return {
#             "message": "Response Produced Successfully",
#             "prediction": decoded
#         }
    # src/inference/app.py
#!/usr/bin/env python3
import sys
import os
import pickle
from dotenv import load_dotenv
import numpy as np
import pandas as pd
import mlflow
import dvc.api
import dagshub
from types import SimpleNamespace
from src.logger import ExecutorLogger
from src.training.model_wrapper import ModelWrapper
import litserve as ls

# Utility to convert dicts to namespaces

def dict_to_namespace(d):
    for k, v in d.items():
        if isinstance(v, dict):
            d[k] = dict_to_namespace(v)
    return SimpleNamespace(**d)

class InferenceAPI(ls.LitAPI):
    max_batch_size = 1
    enable_async = False

    def setup(self, device: str = "cpu"):
        load_dotenv()
        # obtain run_id from env or fallback
        self.run_id = os.getenv("RUN_ID")
        if not self.run_id:
            raise ValueError("RUN_ID environment variable must be set.")

        # load configuration
        params = dvc.api.params_show("params.yaml")
        model_cfg = dict_to_namespace(params["model"])

        # init DagsHub & MLflow
        dagshub.auth.add_app_token(os.getenv("DAGSHUB_TOKEN"))
        dagshub.init(
            repo_owner=os.getenv("DAGSHUB_USERNAME"),
            repo_name=model_cfg.repo_name,
            mlflow=model_cfg.use_mlflow,
        )
        mlflow.set_tracking_uri(model_cfg.tracking_uri)
        self.logger = ExecutorLogger("inference-api")
        self.logger.info(f"Loading model run_id={self.run_id}")

        # load pyfunc model
        model_uri = f"runs:/{self.run_id}/model"
        self._model = mlflow.pyfunc.load_model(model_uri)

        # load decoder mapping
        model_dir = os.path.join(model_cfg.model_path, model_cfg.model_name)
        with open(os.path.join(model_dir, "model_target_translator.pkl"), "rb") as f:
            trans = pickle.load(f)
        self._decoder = trans["decoder"]
        self.logger.info("Model and decoder loaded")

    def decode_request(self, request: dict):
        data = list(request.get("input", {}).values())
        arr = np.array(data)[None, :]
        return arr

    def predict(self, x: np.ndarray):
        if x is None:
            return None
        return self._model.predict(x)

    def encode_response(self, output):
        if output is None:
            return {"message": "Error Occurred", "prediction": None}
        decoded = [self._decoder[int(v)] for v in output]
        return {"message": "Response Produced Successfully", "prediction": decoded}

