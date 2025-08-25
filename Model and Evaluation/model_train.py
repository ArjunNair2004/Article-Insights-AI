import os
import logging
import joblib
import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import RidgeCV
from sklearn.ensemble import StackingRegressor
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score, recall_score
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from lightgbm import LGBMClassifier, LGBMRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from imblearn.over_sampling import SMOTE

from website.utils.log_share_wrapper import LogShareWrapper

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

LGB_GPU_PLATFORM_ID = 1
LGB_GPU_DEVICE_ID = 0

def tune_model(trial, model_type):
    if model_type == "lgbm":
        return LGBMRegressor(
            n_estimators=trial.suggest_int("n_estimators", 100, 1000),
            learning_rate=trial.suggest_float("learning_rate", 0.01, 0.2),
            max_depth=trial.suggest_int("max_depth", 5, 12),
            num_leaves=trial.suggest_int("num_leaves", 31, 150),
            random_state=42,
            device='gpu',
            gpu_platform_id=LGB_GPU_PLATFORM_ID,
            gpu_device_id=LGB_GPU_DEVICE_ID,
        )
    elif model_type == "xgb":
        return XGBRegressor(
            n_estimators=trial.suggest_int("n_estimators", 100, 1000),
            learning_rate=trial.suggest_float("learning_rate", 0.01, 0.2),
            max_depth=trial.suggest_int("max_depth", 4, 10),
            subsample=trial.suggest_float("subsample", 0.7, 1.0),
            colsample_bytree=trial.suggest_float("colsample_bytree", 0.7, 1.0),
            verbosity=0,
            random_state=42,
            tree_method='gpu_hist',
            predictor='gpu_predictor',
            gpu_id=0
        )
    elif model_type == "cat":
        return CatBoostRegressor(
            iterations=trial.suggest_int("iterations", 50, 400),
            learning_rate=trial.suggest_float("learning_rate", 0.01, 0.2),
            depth=trial.suggest_int("depth", 4, 8),
            verbose=0,
            random_state=42,
            task_type="GPU",
            devices="0",
            allow_writing_files=False,
            thread_count=1
        )

def optimize(model_type, X_train, y_train, X_valid, y_valid):
    import optuna
    def objective(trial):
        model = tune_model(trial, model_type)
        model.fit(X_train, y_train)
        preds = model.predict(X_valid)
        return mean_absolute_error(y_valid, preds)
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=40)
    logger.info(f"Best {model_type} params: {study.best_params}")
    model = tune_model(study.best_trial, model_type)
    model.fit(X_train, y_train)
    return model

def main():
    os.makedirs("model", exist_ok=True)
    df = pd.read_csv("dataset/final_dataset.csv")
    if "shares" not in df.columns:
        raise ValueError("Missing 'shares' column in dataset")

    virality_threshold = df["shares"].quantile(0.8)
    df["is_viral"] = (df["shares"] >= virality_threshold).astype(int)
    logger.info(f"Viral cutoff (80th percentile): {virality_threshold:.2f}")
    logger.info(f"Class balance:\n{df['is_viral'].value_counts(normalize=True)}")

    drop_cols = ["shares", "log_shares", "is_viral", "style_name", "style_description", "style_emoji", "virality_level"]
    X = df.drop(columns=[col for col in drop_cols if col in df.columns])
    y_reg = np.log1p(df["shares"])
    y_clf = df["is_viral"]

    feature_columns = X.columns.tolist()
    joblib.dump(feature_columns, "model/feature_columns.pkl")

    X_train, X_val, y_train_reg, y_val_reg = train_test_split(X, y_reg, test_size=0.2, random_state=42)
    y_train_clf = y_clf.loc[X_train.index]
    y_val_clf = y_clf.loc[X_val.index]

    numeric_features = X.select_dtypes(include=["number"]).columns.tolist()
    categorical_features = X.select_dtypes(include=["object"]).columns.tolist()

    numeric_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="mean")),
        ("scale", StandardScaler()),
    ])
    if categorical_features:
        categorical_transformer = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore"))
        ])
    else:
        categorical_transformer = None
    transformers = [("num", numeric_transformer, numeric_features)]
    if categorical_transformer:
        transformers.append(("cat", categorical_transformer, categorical_features))
    preprocessor = ColumnTransformer(transformers)
    X_train_proc = preprocessor.fit_transform(X_train)
    X_val_proc = preprocessor.transform(X_val)

    sw = np.where(np.expm1(y_train_reg) > 3000, 5, 1)

    lgbm = optimize("lgbm", X_train_proc, y_train_reg, X_val_proc, y_val_reg)
    xgb  = optimize("xgb", X_train_proc, y_train_reg, X_val_proc, y_val_reg)
    cat  = optimize("cat", X_train_proc, y_train_reg, X_val_proc, y_val_reg)

    stack = StackingRegressor(
        estimators=[("lgbm", lgbm), ("xgb", xgb), ("cat", cat)],
        final_estimator=RidgeCV(),
        passthrough=True,
        n_jobs=1
    )
    stack.fit(X_train_proc, y_train_reg, sample_weight=sw)

    y_pred_log = stack.predict(X_val_proc)
    y_pred_orig = np.expm1(y_pred_log)
    y_val_orig = np.expm1(y_val_reg)
    logger.info(f"MAE: {mean_absolute_error(y_val_orig, y_pred_orig):.1f}, R²: {r2_score(y_val_orig, y_pred_orig):.3f}")
    logger.info(f"Predicted SHARES (max): {y_pred_orig.max()}, (true max): {y_val_orig.max()}")

    smote = SMOTE(random_state=42)
    X_train_clf_sm, y_train_clf_sm = smote.fit_resample(X_train_proc, y_train_clf)
    clf_base = LGBMClassifier(
        n_estimators=500, max_depth=10, learning_rate=0.05, is_unbalance=True, random_state=42,
        device='gpu',
        gpu_platform_id=LGB_GPU_PLATFORM_ID,
        gpu_device_id=LGB_GPU_DEVICE_ID,
    )
    clf = CalibratedClassifierCV(clf_base, method="sigmoid", cv=3)
    clf.fit(X_train_clf_sm, y_train_clf_sm)

    y_proba = clf.predict_proba(X_val_proc)[:, 1]
    thresholds = np.linspace(0.1, 0.9, 100)
    recalls = [recall_score(y_val_clf, (y_proba >= t).astype(int)) for t in thresholds]
    best_thresh = thresholds[np.argmax(recalls)]
    logger.info(f"Best classifier threshold (recall opt): {best_thresh:.2f}")

    sharp_cluster_features = [
        'n_tokens_content', 'num_imgs', 'num_videos',
        'media_richness', 'keyword_density', 'self_reference_min_shares'
    ]
    cluster_data = df[sharp_cluster_features].fillna(0)
    scaler = StandardScaler()
    cluster_scaled = scaler.fit_transform(cluster_data)
    best_k, best_score = 2, -1
    for k in range(2, 10):
        kmeans_temp = KMeans(n_clusters=k, random_state=42, n_init=10)
        score = silhouette_score(cluster_scaled, kmeans_temp.fit_predict(cluster_scaled))
        if score > best_score:
            best_k, best_score = k, score
    kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
    kmeans.fit(cluster_scaled)

    joblib.dump({
        "reg_model": LogShareWrapper(stack),
        "clf_model": clf,
        "preprocessor": preprocessor,
        "virality_threshold": virality_threshold,  
        "classification_threshold": best_thresh,
        "cluster_model": kmeans,
        "cluster_scaler": scaler,
        "cluster_features": sharp_cluster_features,
        "feature_columns": feature_columns
    }, "model/news_model_bundle.pkl")
    logger.info("Training and bundling completed successfully.")

if __name__ == "__main__":
    main()
