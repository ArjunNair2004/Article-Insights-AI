import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import warnings


warnings.filterwarnings("ignore")


def compute_content_quality(row):
    score = 100
    if row.get("n_unique_tokens", 1) < 0.2:
        score -= 10
    if row.get("global_subjectivity", 0) > 0.5:
        score -= 10
    if row.get("avg_negative_polarity", 0) > 0.4:
        score -= 10
    if row.get("n_non_stop_unique_tokens", 1) < 0.3:
        score -= 10
    return max(score, 0)


def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df.drop(columns=[col for col in ['url', 'timedelta', 'title_len', 'title_readability', 'n_tokens_title'] if col in df.columns], inplace=True)

    # --- Virality fields ---
    if 'shares' in df.columns:
        df['log_shares'] = np.log1p(df['shares'])
        df['is_viral'] = (df['shares'] >= 5000).astype(int)
        if len(df) > 10 and df['shares'].nunique() > 2:
            df['virality_level'] = pd.qcut(df['shares'], q=3, labels=['low', 'medium', 'high'])

    if 'virality_level' not in df.columns:
        df['virality_level'] = 'medium'

    epsilon = 1e-5
    df['img_to_word_ratio'] = np.log1p(df.get('num_imgs', 0) / (df.get('n_tokens_content', 1) + epsilon))
    df['video_to_word_ratio'] = np.log1p(df.get('num_videos', 0) / (df.get('n_tokens_content', 1) + epsilon))
    df['media_richness'] = df.get('num_imgs', 0) + df.get('num_videos', 0)

    if 'weekday_is_saturday' in df.columns and 'weekday_is_sunday' in df.columns:
        df['is_weekend'] = df[['weekday_is_saturday', 'weekday_is_sunday']].max(axis=1)
    else:
        df['is_weekend'] = df.get('is_weekend', 0)

    category_cols = [col for col in df.columns if col.startswith('data_channel_is_')]
    if category_cols:
        df['content_category'] = df[category_cols].idxmax(axis=1).str.replace('data_channel_is_', '', regex=False)
    else:
        df['content_category'] = df.get('content_category', 'unknown')

    keyword_cols = [col for col in df.columns if col.startswith('kw_')]
    if keyword_cols:
        df['primary_keyword'] = df[keyword_cols].idxmax(axis=1).str.replace('kw_', '', regex=False)
        df['keyword_density'] = df[keyword_cols].sum(axis=1) / (df.get('n_tokens_content', 1) + 1)
    else:
        df['primary_keyword'] = df.get('primary_keyword', 'none')
        df['keyword_density'] = 0

    np.random.seed(42)
    if 'post_hour' not in df.columns:
        df['post_hour'] = np.random.randint(0, 24, size=len(df))
    if 'post_dayofweek' not in df.columns:
        df['post_dayofweek'] = np.random.randint(0, 7, size=len(df))

    df['subjectivity_sentiment'] = df.get('global_subjectivity', 0) * df.get('global_sentiment_polarity', 0)
    df['positive_sentiment_strength'] = df.get('global_sentiment_polarity', 0) * df.get('avg_positive_polarity', 0)
    df['negative_sentiment_strength'] = df.get('global_sentiment_polarity', 0) * df.get('avg_negative_polarity', 0)
    df['lexical_diversity'] = df.get('n_unique_tokens', 0) / (df.get('n_tokens_content', 1) + 1e-5)

    df['content_quality_score'] = df.apply(compute_content_quality, axis=1)

    # --- Cluster features and fallback clustering
    cluster_features = [
        'global_subjectivity', 'global_sentiment_polarity',
        'avg_positive_polarity', 'avg_negative_polarity',
        'img_to_word_ratio', 'video_to_word_ratio',
        'is_weekend', 'post_hour'
    ]
    for feature in cluster_features:
        if feature not in df.columns:
            df[feature] = 0

    if len(df) < 7:
        df['content_style_cluster'] = 0
    else:
        cluster_data = df[cluster_features].fillna(0)
        scaler = StandardScaler()
        cluster_scaled = scaler.fit_transform(cluster_data)
        final_kmeans = KMeans(n_clusters=7, n_init=10, random_state=42)
        df['content_style_cluster'] = final_kmeans.fit_predict(cluster_scaled)

    cluster_mapping = {
        0: ("Balanced Media Article", "Moderate length with several images—general, versatile content."),
        1: ("Promo/Branded Article", "Slightly shorter, promotional, with a mix of images and video."),
        2: ("Data Error/Artifact", "Data problem/outlier—ignore for normal use."),
        3: ("Standard Feature", "Medium length, moderate media—bread & butter content style."),
        4: ("Extreme Visual Showcase", "Very long, extremely heavy on images—possible gallery/article showcase."),
        5: ("In-Depth Mega Feature", "Very long with high images and videos—deep, media-rich content."),
        6: ("Personal Story/Tutorial", "Moderate articles with lots of video and very high self-reference—likely experience-driven or personal."),
    }

    df['style_name'] = df['content_style_cluster'].map(lambda x: cluster_mapping.get(x, ("Unknown", "Unmapped style", ""))[0])
    df['style_description'] = df['content_style_cluster'].map(lambda x: cluster_mapping.get(x, ("Unknown", "Unmapped style", ""))[1])
    df['style_emoji'] = ""

    df['average_token_length'] = 0
    df['title_subjectivity'] = df.get('global_subjectivity', 0.4)
    df['abs_title_subjectivity'] = abs(df['title_subjectivity'])
    df['title_sentiment_polarity'] = df.get('global_sentiment_polarity', 0.1)
    df['abs_title_sentiment_polarity'] = abs(df['title_sentiment_polarity'])

    df['kw_avg_max'] = 0.1
    df['kw_avg_min'] = 0.01
    df['kw_max_avg'] = 0.2
    df['kw_max_max'] = 0.3
    df['kw_max_min'] = 0.15
    df['kw_min_avg'] = 0.05
    df['kw_min_max'] = 0.1
    df['kw_min_min'] = 0.01

    df['min_negative_polarity'] = -0.2
    df['max_negative_polarity'] = -0.1
    df['min_positive_polarity'] = 0.1
    df['max_positive_polarity'] = 0.4

    df['rate_positive_words'] = 0.2
    df['rate_negative_words'] = 0.1
    df['global_rate_positive_words'] = 0.25
    df['global_rate_negative_words'] = 0.1

    df['n_non_stop_words'] = 0.7
    df['num_self_hrefs'] = df.get('num_hrefs', 2)

    df['LDA_00'] = 0.2
    df['LDA_01'] = 0.2
    df['LDA_02'] = 0.2
    df['LDA_03'] = 0.2
    df['LDA_04'] = 0.2

    return df


if __name__ == "__main__":
    input_path = "dataset/cleaned_dataset.csv"
    output_path = "dataset/final_dataset.csv"

    os.makedirs("dataset", exist_ok=True)
    df = pd.read_csv(input_path)
    df_fe = feature_engineering(df)
    df_fe.to_csv(output_path, index=False)

    print(f"Feature engineering complete. Output saved to: {output_path}")
