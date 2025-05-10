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
from src.inference.requests import InferenceRequest
# Remove the local redefinition of dtype_conversion!
# Keep only the correct import:
from src.training.process_data import dtype_conversion

# Inject into __main__ and __mp_main__
import __main__
__main__.dtype_conversion = dtype_conversion

# Handle __mp_main__ for multiprocessing
import sys
if "__mp_main__" in sys.modules:
    sys.modules["__mp_main__"].dtype_conversion = dtype_conversion
# Utility to convert dicts to namespaces
def dict_to_namespace(d):
    for k, v in d.items():
        if isinstance(v, dict):
            d[k] = dict_to_namespace(v)
    return SimpleNamespace(**d)
# Get the absolute path to the project root
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

# Construct full paths to the pickle files
MODEL_PATH = os.path.join(PROJECT_ROOT, 'models', 'basemodel', 'final_model.pkl')
TRANSLATOR_PATH = os.path.join(PROJECT_ROOT, 'models', 'basemodel', 'model_target_translator.pkl')
PIPELINE_PATH    = os.path.join(PROJECT_ROOT, "data", "processed", "pipeline.pkl")
import joblib


class InferenceAPI(ls.LitAPI):        
    def setup(self, device = "cpu"):
        with open(MODEL_PATH,'rb')as pkl:
            self._model = pickle.load(pkl)
        with open(TRANSLATOR_PATH,'rb')as pkl:
            self._translator = pickle.load(pkl)
        with  open(PIPELINE_PATH,'rb') as pkl:
            self._pipeline  = joblib.load(pkl)
        self._raw_cols = [
                "PassengerId", "Survived", "Pclass", "Name", "Sex", 
                "Age", "SibSp", "Parch", "Ticket", "Fare", "Cabin", "Embarked"
            ]
        self._translator["decoder"] = {0: "did_not_survive", 1: "survived"}


    def decode_request(self, request: dict):
        """
        Expects JSON: {"input": {col1: val1, col2: val2, ...}}
        Returns: pd.DataFrame of shape (1, n_raw_cols)
        """
        data = request.get("input", {})
        row = {col: data.get(col, None) for col in self._raw_cols}
        return pd.DataFrame([row], columns=self._raw_cols)

    def predict(self, df: pd.DataFrame):
        if df is None:
            return None
        X = self._pipeline.transform(df)
        # if it ended up still having Survived:
        if "Survived" in X.columns:
            X = X.drop(columns="Survived")
        return self._model.predict(X)



    def encode_response(self, output):
        if output is None:
            message = "Error Occurred"
        else:
            message = "Response Produced Successfully"
        return {
            "message": message,
            "prediction": [self._translator['decoder'][val] for val in output]
        }
