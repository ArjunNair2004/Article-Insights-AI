import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import base64
from io import BytesIO

sns.set_style("whitegrid")

# --- 1. Best Time to Publish Heatmap ---
def generate_best_time_heatmap():
    days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    hours = list(range(0, 24))
    data = np.random.rand(7, 24)  
    df = pd.DataFrame(data, index=days, columns=hours)

    plt.figure(figsize=(12, 6), dpi=150)
    ax = sns.heatmap(df, cmap="YlGnBu", cbar_kws={'label': 'Engagement Score'})
    plt.title(" Best Times to Publish (Lighter = Better)", fontsize=16)
    plt.xlabel("Hour of Day")
    plt.ylabel("Day of Week")
    plt.tight_layout()

    return _encode_plot_to_base64(), "Lighter blocks show better times to post for higher engagement."

# --- 2. Channel Effectiveness Plot ---
def generate_channel_effectiveness_plot():
    data = {
        'Channel': ['Social', 'Email', 'Organic', 'Referral'],
        'Effectiveness': [78, 65, 85, 55]
    }
    df = pd.DataFrame(data)

    plt.figure(figsize=(10, 6), dpi=150)
    ax = sns.barplot(x='Channel', y='Effectiveness', data=df, palette='Set2')
    ax.bar_label(ax.containers[0], fmt='%d%%', padding=5)
    plt.title(" Channel Effectiveness", fontsize=16)
    plt.ylabel("Engagement (%)")
    plt.tight_layout()

    return _encode_plot_to_base64(), "This shows how well each channel performs in driving shares."

# --- 3. Media Impact Plot ---
def generate_media_impact_plot():
    categories = ['No Media', 'Images', 'Videos', 'Images + Videos']
    impact = [2000, 3500, 3000, 5000]

    plt.figure(figsize=(10, 6), dpi=150)
    ax = sns.barplot(x=categories, y=impact, palette='coolwarm')
    ax.bar_label(ax.containers[0], fmt='%d', padding=5)
    plt.title(" Media Types vs. Average Shares", fontsize=16)
    plt.ylabel("Average Shares")
    plt.tight_layout()

    return _encode_plot_to_base64(), "Combining images and videos tends to get the most shares."

# --- 4. Content Quality Score & Suggestions ---
def compute_quality_score_and_suggestions(df):
    score = 0
    suggestions = []
    breakdown = []

    content_length = df['n_tokens_content'].iloc[0]
    num_imgs = df['num_imgs'].iloc[0]
    num_vids = df['num_videos'].iloc[0]
    keyword_density = df['num_keywords'].iloc[0]
    sentiment = df.get('global_sentiment_polarity', pd.Series([0])).iloc[0]
    ref_count = df.get('self_reference_min_shares', pd.Series([0])).iloc[0]

    if content_length >= 500:
        score += 1
        breakdown.append(("Content Length", 1))
    else:
        suggestions.append("Consider increasing content length for better engagement.")
        breakdown.append(("Content Length", 0))

    if num_imgs >= 1:
        score += 1
        breakdown.append(("Images Present", 1))
    else:
        suggestions.append("Add at least one relevant image.")
        breakdown.append(("Images Present", 0))

    if num_vids >= 1:
        score += 1
        breakdown.append(("Videos Present", 1))
    else:
        suggestions.append("Try embedding a video to retain attention.")
        breakdown.append(("Videos Present", 0))

    if keyword_density >= 0.1:
        score += 1
        breakdown.append(("Keyword Usage Density", 1))
    else:
        suggestions.append("Use more specific and relevant keywords.")
        breakdown.append(("Keyword Usage Density", 0))

    if sentiment < 0:
        suggestions.append("Try using a more positive tone.")
        breakdown.append(("Positive Tone", 0))
    elif sentiment > 0.5:
        score += 0.5
        breakdown.append(("Positive Tone", 0.5))
    else:
        breakdown.append(("Positive Tone", 0))

    if ref_count >= 3:
        score += 0.5
        breakdown.append(("Reference Count", 0.5))
    else:
        suggestions.append("Reference more authoritative or previously successful articles.")
        breakdown.append(("Reference Count", 0))

    final_score = round(score * 20)

    if final_score >= 80:
        viral_status = "Likely Viral"
    elif final_score >= 60:
        viral_status = "Possibly Viral"
    elif final_score >= 40:
        viral_status = "Unlikely Viral"
    else:
        viral_status = "Poor Content"

    return final_score, breakdown, viral_status, suggestions

# --- 5. Helper for Plot Encoding ---
def _encode_plot_to_base64():
    buffer = BytesIO()
    fig = plt.gcf()
    fig.savefig(buffer, format='png', bbox_inches='tight')
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
    plt.close(fig)
    return f"data:image/png;base64,{image_base64}"

# --- 6. Content Style Cluster Output ---
def get_cluster_style_output(cluster_id: int):
    cluster_mapping = {
        0: ("Balanced Media Article",      "Moderate length with several images—general, versatile content."),
        1: ("Promo/Branded Article",       "Slightly shorter, promotional, with a mix of images and video."),
        2: ("Data Error/Artifact",         "Data problem/outlier—ignore for normal use."),
        3: ("Standard Feature",            "Medium length, moderate media—bread & butter content style."),
        4: ("Extreme Visual Showcase",     "Very long, extremely heavy on images—possible gallery/article showcase."),
        5: ("In-Depth Mega Feature",       "Very long with high images and videos—deep, media-rich content."),
        6: ("Personal Story/Tutorial",     "Moderate articles with lots of video and very high self-reference—likely experience-driven or personal.", "📝"),
    }
    if cluster_id in cluster_mapping:
        name, desc, emoji = cluster_mapping[cluster_id]
        return {
            "cluster_id": cluster_id,
            "style_name": name,
            "description": desc,
            "emoji": emoji
        }
    else:
        return {
            "cluster_id": cluster_id,
            "style_name": "Unknown",
            "description": "Unrecognized content style.",
            "emoji": "❓"
        }
