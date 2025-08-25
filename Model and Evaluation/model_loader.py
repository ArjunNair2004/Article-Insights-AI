import os
import joblib
import sys

import src.website as website_module
sys.modules["website"] = website_module

from .log_share_wrapper import LogShareWrapper

def load_models():
    """
    Load the trained regression model, classification model, preprocessor, 
    thresholds, clustering model, scaler, and feature list for clustering.
    """
    bundle_path = os.path.join("model", "news_model_bundle.pkl")
    bundle = joblib.load(bundle_path)

    reg_model = bundle["reg_model"]
    clf_model = bundle["clf_model"]
    preprocessor = bundle["preprocessor"]
    virality_threshold = bundle["virality_threshold"]
    classification_threshold = bundle["classification_threshold"]
    cluster_model = bundle.get("cluster_model", None)
    cluster_scaler = bundle.get("cluster_scaler", None)
    cluster_features = bundle.get("cluster_features", None)

    return (
        reg_model,
        clf_model,
        preprocessor,
        virality_threshold,
        classification_threshold,
        cluster_model,
        cluster_scaler,
        cluster_features  
    )

