import json
import os
import pickle
from sklearn.metrics import classification_report
from skore import EstimatorReport


def evaluate(X_test, y_test, cfg, logger):
    """Proper evaluation function with correct encoding"""
    logger.info("Starting model evaluation")
    model_name = cfg.model_name
    MODEL_PATH = cfg.model_path
    REPORT_PATH = cfg.report_path
    try:
        # Load artifacts
        with open(
            os.path.join(MODEL_PATH, model_name, "model_target_translator.pkl"), "rb"
        ) as pkl:
            translator = pickle.load(pkl)

        with open(os.path.join(MODEL_PATH, model_name, "final_model.pkl"), "rb") as pkl:
            model = pickle.load(pkl)

        # Encode test labels
        y_test_enc = translator["encoder"].transform(y_test)

        # Generate predictions
        y_pred = model.predict(X_test)

        # Convert numeric class labels to strings
        class_names = [str(v) for v in translator["decoder"].values()]

        # Generate classification report
        evaluation_report = classification_report(
            y_test_enc,
            y_pred,
            target_names=class_names,  # Use string labels
        )

        logger.info("saving evaluation report")
        if not os.path.exists(os.path.join(REPORT_PATH, model_name)):
            os.makedirs(os.path.join(REPORT_PATH, model_name))
        with open(
            os.path.join(REPORT_PATH, model_name, "evaluation_report.json"), "w"
        ) as js:
            json.dump(evaluation_report, js, indent=4)
        logger.info(f"Evaluation Report:\n{evaluation_report}")

    except Exception as e:
        logger.error(f"Evaluation failed: {str(e)}")
        raise
