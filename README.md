Title
Article Insights AI — Predict Shares, Estimate Virality, and Get Actionable Editorial Insights

Overview
Article Insights AI is a FastAPI web application that helps creators and editors forecast article performance before publishing. It predicts expected share counts, estimates virality likelihood, assigns an interpretable content style, and provides practical improvement tips. The app bundles automated feature engineering, a stacked regression model for reach, a calibrated classifier with business‑oriented thresholds, and clustering for style guidance—rendered in a clean, responsive UI with embeddable visual diagnostics.

Key features
Share count prediction: A stacked ensemble trained on log(1+shares) provides stable estimates and returns them on the original scale for planning.
Virality estimation: A calibrated probability coupled with a practical share cutoff reduces false positives for low‑reach content.
Style clustering: Interpretable segments (e.g., Balanced Media, Promo/Branded, In‑Depth) mapped from unsupervised clusters into human‑readable labels.
Quality score and suggestions: Lightweight scoring from length, media, keywords, and tone with clear, actionable tips to improve drafts.
Visual insights: Best‑time heatmap, channel effectiveness, and media impact charts embedded in the results page as images.
Consistent inference: A serialized bundle ships models, preprocessing, thresholds, scaler, and cluster artifacts to keep training and inference aligned.

How it works
Input: Use a unified form to enter article metrics (length, images, videos, keywords, self‑references, category, weekday). Optionally accept pasted title and article content to auto‑derive features via lightweight NLP for a “copy–paste → insights” flow.
Processing: The server reconstructs a full feature vector, fills any missing fields conservatively, executes feature engineering, and applies the same preprocessing used at training time (training–inference parity).

Predictions:
Regression: Returns expected shares after inverse‑transforming the log target.
Classification: Produces a calibrated virality probability and applies a share cutoff to label Viral/Not Viral.
Clustering: Assigns an interpretable content style with a short, descriptive summary.
Output: Results render as cards with optional visuals and suggestions; users select which sections to generate.

Why it’s useful
Reduces editorial guesswork by quantifying likely reach and surfacing high‑leverage improvements before publishing.
Provides pragmatic virality flags that combine probability and expected reach, so promotion focuses on meaningful candidates.
Offers interpretable style guidance and suggestions, helping teams position content quickly and iterate with intent.

Architecture highlights
Web: FastAPI backend, Jinja2 templates, responsive front‑end with consistent navigation and glassmorphic cards.
Models: Stacked regressor for reach, calibrated classifier for virality, and KMeans clustering with scaler and feature list for reproducible assignments.
Artifacts: Single bundle holding model objects, feature schema, thresholds, scaler, and cluster feature names.
Robustness: Mean‑filling for missing fields at inference, safe probability extraction for single‑sample requests, and consistent tuple mapping for cluster metadata.

Getting started

Prerequisites:
Python 3.9+ (recommend 3.10–3.11)
A virtual environment tool (venv, uv, or conda)

Setup:
Create and activate a virtual environment.
Install dependencies: pip install -r requirements.txt
Place trained artifacts (e.g., news_model_bundle.pkl and feature_columns.pkl) into the model/ directory.

Run:
Start the app with Uvicorn (dev): uvicorn src.app:app --reload
Visit the form page to submit an article’s metrics, select desired outputs, and review predictions.

Typical workflow
Open “Unified Form” → enter metrics (or paste title/content if enabled) → select outputs (shares, virality, style, visuals, suggestions) → submit → review predictions, visuals, and improvement tips → iterate and resubmit quickly.

Dataset
Description: The dataset represents article attributes (e.g., tokenized length, media counts, keyword counts, category/channel indicators, weekday) and outcome signals (shares). It underpins training of the share regressor, the virality classifier, and clustering.

Schema highlights:
Core numeric features: n_tokens_content, num_imgs, num_videos, num_keywords, self_reference_min_shares.
Categorical/dummies: content_category (mapped to channel dummies), weekday dummies.
Engineered features (during training): media ratios, keyword density, sentiment/subjectivity interactions, and any timing indicators used in modeling.

Storage strategy:
Do not commit large datasets to the repository to avoid bloating history and hitting file size limits.
Preferred: provide a scripts/fetch_data.py that downloads data to dataset/ and verifies checksums; keep dataset/.gitkeep to retain the folder.
Alternative: use Git LFS for moderate‑sized CSVs if organizationally acceptable; document storage/bandwidth quotas and how to install LFS.

Reproducibility:
Document data version/date, source, and license/usage terms.
Include SHA256 checksums for downloadable files in the fetch script or a checksums.txt to ensure integrity.
Optionally include a tiny sample (e.g., 50–200 rows) for quick sanity checks and for CI without large transfers.

Models and artifacts

Required artifacts:
Model bundle (e.g., news_model_bundle.pkl) containing: regressor, classifier, preprocessor, thresholds, cluster model, scaler, and cluster feature list.
feature_columns.pkl used to align incoming features with the training schema.
Placement: Store artifacts in the model/ directory (ignored by default for large assets if you prefer external hosting).
Regeneration: If training code is included, document how to retrain (data prep → train → evaluate → export bundle) and how to update feature_columns.pkl for compatibility.

Project structure
src/
app.py (or main.py): FastAPI application factory/entry point.

website/
init.py: Declares website as a Python package so imports resolve cleanly; can also set package‑level constants.
routes.py: HTTP route handlers (form, predict, results, static pages).
templates/: Jinja2 templates (home.html, about.html, services.html, unified_form.html, result.html).
utils/
init.py: Declares utils as a package, allowing absolute imports like from src.website.utils import feature_eng.
model_loader.py: Loads serialized bundle and returns components (models, preprocessor, thresholds, clustering artifacts).
feature_eng.py: Feature engineering functions to transform raw inputs into model‑ready features.
web_outputs.py: Plot/visual helpers (heatmap, channel and media impact), quality scoring, and suggestions.

model/: Serialized artifacts (bundle, feature_columns). Consider external hosting or Git LFS for large files.

dataset/: Downloaded data (kept out of Git); include .gitkeep and a fetch script.

scripts/: Diagnostics (e.g., classifier tests) and data fetchers.

docs/: Screenshots and architecture notes.

requirements.txt, README.md, .gitignore, LICENSE, .env.example

Why init.py is used in each package
Purpose: init.py marks a directory as a Python package so absolute imports work (e.g., from src.website.utils import feature_eng). It runs once on package import and can initialize package‑level state, set all to define a public API, or perform light configuration.
In this project:
src/ (optional): If treating src as a package, include init.py to enable from src.website import routes.
src/website/: init.py ensures website is importable as a package, allowing clean imports across routes and utilities.
src/website/utils/: init.py enables utils to be imported as a subpackage and optionally re‑exports common functions for convenience (e.g., from .feature_eng import feature_engineering).
Benefit: Makes module resolution explicit and stable across environments and tooling; helps avoid import path issues when running the app or tests.

Configuration

Environment variables: Provide a .env.example describing variables (e.g., MODEL_BUNDLE_PATH, LOG_LEVEL). Users copy it to .env and fill their values.
Secrets: Never commit real .env files or secrets; ensure .env is listed in .gitignore.

Usage
Web interface: Visit /form, enter article details (or paste title/content if enabled), select outputs, and submit to see predictions and visuals.
API (optional): If exposing JSON endpoints, document routes, query parameters, and example responses for programmatic access.

Testing
Unit tests: Add basic tests for utility functions and feature engineering to ensure schema stability.
Diagnostics: Provide a small script to check classifier classes, feature alignment, and preprocessor transformations on a synthetic sample.

Deployment
Local: uvicorn src.app:app --reload for development.
Docker (optional): Supply Dockerfile and compose for reproducible deployments; document environment variables for container runtime.
Cloud: Mention reverse proxy, TLS termination, and environment config if deploying to a PaaS or VM.

Roadmap
URL ingestion with NLP extraction: Fetch a page by URL, extract main content/title, and auto‑populate features; include language detection and boilerplate removal.
Timed re‑reading (timedeltas): Re‑fetch URLs at configurable intervals to detect updates, refresh predictions, and record change logs.
Richer NLP features: Title readability/salience, keyphrases/entities, and topic/semantic embeddings; retrain once incremental lift is demonstrated.
Real‑time feedback: Use early engagement metrics (impressions, CTR, dwell time) to recalibrate probabilities and thresholds.
Segment‑aware policies: Cluster/category‑specific thresholds and suggestions; personalized guidance for publisher segments.
Optimization policies: Explore bandits/RL for promotion timing and channel allocation under constraints.
Cloud‑native scale and integrations: Containerization, autoscaling, CMS/editor integrations, and organization analytics.
Monitoring and retraining: Calibration/error dashboards, drift detection, scheduled retrains, and guardrails for edge cases.
Privacy and governance: Configurable retention, anonymization, and audit logs for fetched content and inputs.

Contributing
Issues and pull requests are welcome.
Please keep code style consistent, write concise tests for new utilities, and document changes that affect feature schemas or artifacts.

License
Choose and include a license (MIT is common for open projects). Ensure any dataset licenses are respected and documented.

Acknowledgments

Libraries and tools (FastAPI, scikit‑learn ecosystem, plotting libs).

Any dataset sources or inspirations used during development.
