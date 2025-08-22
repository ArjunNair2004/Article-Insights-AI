Article Insights AI is a FastAPI-powered web application that helps creators predict article performance and improve drafts before publishing. It estimates expected share counts, classifies virality likelihood, and generates practical suggestions, all from a simple form or (optionally) pasted title and content. The system combines automated feature engineering with a stacked regression model for reach, a calibrated classifier for virality, and interpretable content-style clustering, wrapped in a clean, responsive UI.

Core capabilities
Share Count Prediction: A stacked ensemble trained on log(1+shares) returns stable, de-skewed estimates in the original scale for realistic planning.
Virality Classification: A calibrated probability plus a practical share cutoff yields conservative, business-aligned “Viral/Not Viral” labels.
Content Style Clustering: Assigns an interpretable style (e.g., Balanced Media, Promo, In‑Depth) to guide structure and media choices.
Quality Score & Suggestions: Heuristic scoring from length, images/videos, keyword density, tone, and references, with clear, actionable tips.
Visual Insights: Best-time heatmap, channel effectiveness, and media impact charts returned as base64 images for instant embedding in the UI.

How it works
Input: Provide article metrics (length, media counts, keywords, category, weekday), or paste a title and full content to auto-derive features via lightweight NLP. Users can override estimates before running predictions.
Processing: The app applies the same feature engineering and preprocessing used in training to ensure training–inference parity.
Predictions: The regressor outputs expected shares; the classifier provides a calibrated virality probability used with a share cutoff; clustering maps the content to a human-readable style.
Output: Results are rendered in a glassmorphic UI with optional visuals and a suggestions panel, plus flags to show only the sections the user selected.

Why it’s useful
Reduces editorial guesswork by quantifying likely reach and highlighting improvement levers (media richness, timing, structure).
Provides pragmatic virality flags that balance ambition with precision, helping teams prioritize promotion.
Offers interpretable style guidance for quick positioning and experimentation.
Architecture highlights
End-to-end bundle with models, thresholds, feature schema, and preprocessing to keep environments consistent.
Robust request handling with mean-filling for missing fields and safe probability extraction.
Modular utilities for visualization (heatmap/graphs) and content-quality scoring, enabling easy UI composition.

Planned enhancements
URL ingestion with NLP extraction to auto-populate features from live articles, plus timedelta-based re-reading to detect updates and refresh predictions.
Richer NLP features (readability, keyphrases/entities, topic embeddings) with retraining for incremental lift.
Real-time feedback loops (early engagement signals), segment-aware thresholds, and reinforcement learning for promotion timing and channel allocation.
Cloud-native scaling, CMS integrations, monitoring, drift detection, and automated retrains with governance controls.
In short, Article Insights AI is an editorial intelligence layer: fast, interpretable, and production-ready, built to help teams ship higher-impact content with fewer iterations.
