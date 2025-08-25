from fastapi import APIRouter, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import pandas as pd
import joblib
import numpy as np
from typing import List

from src.website.utils.model_loader import load_models
from src.website.utils.web_outputs import (
    generate_best_time_heatmap,
    generate_channel_effectiveness_plot,
    generate_media_impact_plot,
    compute_quality_score_and_suggestions
)
from src.website.utils.feature_eng import feature_engineering

router = APIRouter()
templates = Jinja2Templates(directory="src/website/templates")

# ---------- Load models and supporting items ---------
(
    reg_model,
    clf_model,
    preprocessor,
    virality_threshold,
    classification_threshold,
    cluster_model,
    cluster_scaler,
    cluster_features
) = load_models()

feature_columns = joblib.load("model/feature_columns.pkl")
train_df = pd.read_csv("dataset/final_dataset.csv")
train_means = train_df[feature_columns].mean(numeric_only=True).to_dict()

# -------------- MAIN ROUTES -----------------

@router.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("home.html", {"request": request})

@router.get("/about", response_class=HTMLResponse)
async def about(request: Request):
    return templates.TemplateResponse("about.html", {"request": request})

@router.get("/services", response_class=HTMLResponse)
async def services(request: Request):
    return templates.TemplateResponse("services.html", {"request": request})

@router.get("/form", response_class=HTMLResponse)
async def unified_form(request: Request):
    return templates.TemplateResponse("unified_form.html", {"request": request})

# --------- Unified prediction handler ---------
@router.post("/predict", response_class=HTMLResponse)
async def predict(
    request: Request,
    selected_services: List[str] = Form(...),
    n_tokens_content: int = Form(...),
    num_imgs: int = Form(...),
    num_videos: int = Form(...),
    num_keywords: float = Form(...),
    self_reference_min_shares: int = Form(...),
    content_category: str = Form(...),
    weekday: str = Form(...)
):
    input_data = {
        'n_tokens_content': n_tokens_content,
        'num_imgs': num_imgs,
        'num_videos': num_videos,
        'num_keywords': num_keywords,
        'self_reference_min_shares': self_reference_min_shares,
        'content_category': content_category
    }

    # Set weekday dummy variable
    input_data[f'weekday_is_{weekday.lower()}'] = 1

    # Set channel dummy variables
    channel_map = {
        "tech": "data_channel_is_tech",
        "health": "data_channel_is_lifestyle",
        "lifestyle": "data_channel_is_lifestyle",
        "finance": "data_channel_is_bus",
        "bus": "data_channel_is_bus",
        "entertainment": "data_channel_is_entertainment",
        "socmed": "data_channel_is_socmed",
        "world": "data_channel_is_world",
        "other": "data_channel_is_world"
    }
    for v in [
        "data_channel_is_tech", "data_channel_is_lifestyle", "data_channel_is_bus",
        "data_channel_is_entertainment", "data_channel_is_socmed", "data_channel_is_world"
    ]:
        input_data[v] = 0
    chosen_channel = channel_map.get(content_category.lower(), "data_channel_is_world")
    input_data[chosen_channel] = 1

    # Compose DataFrame and fill missing features
    raw_df = pd.DataFrame([input_data])
    for col in feature_columns:
        if col not in raw_df.columns:
            raw_df[col] = train_means.get(col, 0)
    raw_df = raw_df[feature_columns]

    # Feature engineering pipeline
    df = feature_engineering(raw_df)
    X = preprocessor.transform(df)

    # Prediction logic
    log_share_pred = reg_model.predict(X)[0]
    shares_pred = int(np.expm1(log_share_pred))
    viral_cut = int(np.expm1(virality_threshold))

    # Safe extraction of positive-class probability
    probas = clf_model.predict_proba(X)
    if hasattr(clf_model, "classes_") and len(getattr(clf_model, "classes_", [])) == 2:
        pos_idx = list(clf_model.classes_).index(1)
        proba = float(probas[0][pos_idx])
    else:
        proba = 0.0

    cthresh = max(classification_threshold, 0.18)
    viral_label = "Viral" if (proba >= cthresh and shares_pred >= viral_cut) else "Not Viral"

    # Cluster/style prediction
    cluster_label = None
    cluster_style_name = None
    cluster_description = None
    cluster_icon = None
    if "cluster" in selected_services:
        try:
            cluster_input = df[cluster_features].fillna(0)
            cluster_scaled = cluster_scaler.transform(cluster_input)
            cluster_label = int(cluster_model.predict(cluster_scaled)[0])
        except Exception as e:
            print(f"Cluster prediction error: {e}")
        cluster_styles = {
            0: ("Balanced Media Article", "Moderate length with several images—general, versatile content.", ""),
            1: ("Promo/Branded Article", "Slightly shorter, promotional, with a mix of images and video.", ""),
            2: ("Data Error/Artifact", "Data problem/outlier—ignore for normal use.", ""),
            3: ("Standard Feature", "Medium length, moderate media—bread & butter content style.", ""),
            4: ("Extreme Visual Showcase", "Very long, extremely heavy on images—possible gallery/article showcase.", ""),
            5: ("In-Depth Mega Feature", "Very long with high images and videos—deep, media-rich content.", ""),
            6: ("Personal Story/Tutorial", "Moderate articles with lots of video and very high self-reference—likely experience-driven or personal.", "")
        }
        style = cluster_styles.get(
            cluster_label, ("Unknown", "No description available.", "")
        )
        cluster_style_name, cluster_description, cluster_icon = style

    pub_time_plot = channel_plot = media_plot = None
    pub_time_caption = channel_caption = media_caption = ""
    if "heatmap" in selected_services:
        pub_time_plot, pub_time_caption = generate_best_time_heatmap()
        channel_plot, channel_caption = generate_channel_effectiveness_plot()
        media_plot, media_caption = generate_media_impact_plot()
    quality_score = None
    suggestions = []
    if "suggestions" in selected_services:
        quality_score, _, _, suggestions = compute_quality_score_and_suggestions(df)
        if quality_score == 100:
            suggestions = ["No further improvements needed!"]

    return templates.TemplateResponse("result.html", {
        "request": request,
        "selected_services": selected_services,
        "shares_pred": shares_pred,
        "viral_label": viral_label,
        "pub_time_plot": pub_time_plot,
        "channel_plot": channel_plot,
        "media_plot": media_plot,
        "pub_time_caption": pub_time_caption,
        "channel_caption": channel_caption,
        "media_caption": media_caption,
        "content_quality_score": quality_score,
        "suggestions": suggestions,
        "cluster_style_name": cluster_style_name,
        "cluster_description": cluster_description,
        "cluster_icon": cluster_icon,
        "show_share_count": "share_count" in selected_services,
        "show_virality": "virality" in selected_services,
        "show_heatmap": "heatmap" in selected_services,
        "show_suggestions": "suggestions" in selected_services,
        "show_cluster": "cluster" in selected_services
    })
