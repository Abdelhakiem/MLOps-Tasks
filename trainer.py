from src.logger import ExecutorLogger
from src.training.process_data import read_process_data
from src.training.evaluate import evaluate
from src.training.train import train_model,encode_target

import hydra
from omegaconf import DictConfig, OmegaConf
@hydra.main(config_path="conf", config_name="config", version_base=None)
def train_pipeline(cfg: DictConfig):
    logger = ExecutorLogger('train pipeline')
    logger.info("Training Started...")
    logger.info("Pipeline Parameters: \n" f"{OmegaConf.to_yaml(cfg)}")
    read_process_data(cfg.pipeline.data, logger)
    X_train, y_train, X_test, y_test = encode_target( cfg.pipeline.model, logger = logger)
    
    train_model(X = X_train, y = y_train, cfg = cfg.pipeline.model,logger = logger)
    evaluate(X_test, y_test, cfg.pipeline.evaluate, logger)
    logger.info("Training Completed...")
if __name__ == "__main__":
    train_pipeline()