import json
import os
import pickle
import pandas as pd
from sklearn.metrics import classification_report

def evaluate(cfg, logger):
    """Proper evaluation function with correct encoding"""
    logger.info("Starting model evaluation")

    model_name = cfg.model_name
    MODEL_PATH = cfg.model_path
    REPORT_PATH = cfg.report_path
    PROCESSED_PATH = cfg.processed_data_path
    FILE_NAME = cfg.file_name
    TARGET = cfg.target

    try:
        # 1. Load test data
        df_test = pd.read_parquet(os.path.join(PROCESSED_PATH, f"{FILE_NAME}-test.parquet"))
        X_test, y_test = df_test.drop(columns=[TARGET]), df_test[TARGET]

        # 2. Load encoder and model
        with open(os.path.join(MODEL_PATH, model_name, "model_target_translator.pkl"), "rb") as pkl:
            translator = pickle.load(pkl)
        with open(os.path.join(MODEL_PATH, model_name, "final_model.pkl"), "rb") as pkl:
            model = pickle.load(pkl)

        # 3. Encode test labels
        y_test_enc = translator["encoder"].transform(y_test)

        # 4. Predict
        y_pred = model.predict(X_test)

        # 5. Generate classification report (as dict)
        class_names = [str(v) for v in translator["decoder"].values()]
        evaluation_report = classification_report(
            y_test_enc,
            y_pred,
            target_names=class_names,
            output_dict=True  # <-- important for JSON
        )

        # 6. Save report
        os.makedirs(os.path.join(REPORT_PATH, model_name), exist_ok=True)
        with open(os.path.join(REPORT_PATH, model_name, "evaluation_report.json"), "w") as js:
            json.dump(evaluation_report, js, indent=4)

        logger.info(f"Evaluation completed.\nReport:\n{json.dumps(evaluation_report, indent=2)}")

    except Exception as e:
        logger.error(f"Evaluation failed: {str(e)}")
        raise


# Entry point
from omegaconf import OmegaConf
import dvc.api
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.logger import ExecutorLogger
from types import SimpleNamespace

def dict_to_namespace(d):
    for k, v in d.items():
        if isinstance(v, dict):
            d[k] = dict_to_namespace(v)
    return SimpleNamespace(**d)

if __name__ == "__main__":
    cfg = dvc.api.params_show()
    cfg_model = dict_to_namespace(cfg["evaluate"])
    logger = ExecutorLogger("dvc-evaluate")

    evaluate(cfg_model, logger=logger)
