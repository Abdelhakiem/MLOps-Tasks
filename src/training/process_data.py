import pandas as pd
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from feature_engine.imputation import MeanMedianImputer
from feature_engine.encoding import OneHotEncoder
from feature_engine.selection import DropFeatures
from feature_engine.wrappers import SklearnTransformerWrapper
from sklearn.preprocessing import StandardScaler
from omegaconf import DictConfig

def dtype_conversion(X, cat_cols):
    X = X.copy()
    for col in cat_cols:
        if col in X.columns:
            # Fill NA and convert to category
            X[col] = X[col].fillna('missing').astype('category')
    return X
def read_process_data(
    cfg: DictConfig,
    logger
):
    """Data processing pipeline"""
    logger.info("Starting data processing")
    
    try:
        # 1. Load data
        df = pd.read_csv(os.path.join(cfg.raw_data_path, f'{cfg.file_name}.csv'))
        logger.info(f"Raw data loaded: {df.shape}")
        
        # 2. Validate initial data
        if df[cfg.target].isna().any():
            raise ValueError(f"Target column '{cfg.target}' contains missing values")
        
        # 3. Split data
        train_df, test_df = train_test_split(
            df, test_size=0.2, random_state=42, stratify=df[cfg.target]
        )
        logger.info(f"Train/Test split: {train_df.shape}/{test_df.shape}")

        # 4. Create processing pipeline
        cat_cols = list(cfg.features.categorical)
        num_cols = list(cfg.features.numeric)
        drop_cols = list(cfg.features.drop)
        processing_pipeline = Pipeline([
            ('dtype_conversion', FunctionTransformer(
                func=dtype_conversion,
                kw_args={'cat_cols': cat_cols},
                validate=False
            )),
            
            ('numeric_imputer', MeanMedianImputer(
                imputation_method='median',
                variables=num_cols
            )),
            
            ('encoder', OneHotEncoder(
                drop_last=True,
                variables=cat_cols
            )),
            
            ('scaler', SklearnTransformerWrapper(
                transformer=StandardScaler(),
                variables=num_cols
            )),
            
            ('drop_features', DropFeatures(
                features_to_drop=drop_cols 
            ))
        ])

        # 5. Process data
        X_train = processing_pipeline.fit_transform(train_df)
        X_test = processing_pipeline.transform(test_df)

        # 6. Combine with target (critical fix)
        train_clean = pd.concat([
            X_train,
            train_df[cfg.target].rename(cfg.target)  # Preserve original index
        ], axis=1)
        
        test_clean = pd.concat([
            X_test,
            test_df[cfg.target].rename(cfg.target)  # Preserve original index
        ], axis=1)

        # 7. Validate output
        if len(train_clean) != len(train_df):
            raise ValueError("Row count mismatch in training data")
        if len(test_clean) != len(test_df):
            raise ValueError("Row count mismatch in test data")

        # 8. Save artifacts
        DESTINATION = cfg.processed_data_path
        file_name = cfg.file_name
        os.makedirs(DESTINATION, exist_ok=True)
        train_clean.to_parquet(os.path.join(DESTINATION, f"{file_name}-train.parquet"))
        test_clean.to_parquet(os.path.join(DESTINATION, f"{file_name}-test.parquet"))
        joblib.dump(processing_pipeline, os.path.join(DESTINATION, "pipeline.pkl"))

        logger.info(f"Processing complete. Final shapes: Train {train_clean.shape}, Test {test_clean.shape}")
    except Exception as e:
        logger.error(f"Processing failed: {str(e)}")
        raise e