import streamlit as st
import pandas as pd
import pickle

# -----------------------------
# Load model and scaler
# -----------------------------
with open("model_data.pkl", "rb") as f:
    model_data = pickle.load(f)

kmeans_model = model_data["model"]
scaler = model_data["scaler"]

# -----------------------------
# Load dataset
# -----------------------------
df = pd.read_csv(
    r"C:\Users\IICET 22\Desktop\abhishek\project store\spotify recommendaton system\top 100 streamed_songs.csv"
)

# -----------------------------
# Add cluster labels dynamically
# -----------------------------
try:
    # Drop non-numeric features for clustering
    features_for_clustering = df.drop(columns=["id", "name"], errors="ignore")
    numerical_features = features_for_clustering.select_dtypes(include=["float64", "int64"])

    # Scale features
    scaled_features = scaler.transform(numerical_features)

    # Predict clusters
    cluster_labels = kmeans_model.predict(scaled_features)
    df["cluster_label"] = cluster_labels
except Exception as e:
    st.error(f"⚠️ Could not generate clusters: {e}")
    df["cluster_label"] = -1  # fallback so code doesn't crash

# -----------------------------
# Recommendation function
# -----------------------------
def recommend_songs(song_name, num_recommendations=5):
    if "cluster_label" not in df.columns:
        return []

    if song_name not in df["name"].values:
        return []

    # Find cluster of input song
    song_cluster = df[df["name"] == song_name]["cluster_label"].iloc[0]

    # Find similar songs in same cluster
    similar_songs = df[df["cluster_label"] == song_cluster]
    if similar_songs.empty:
        return []
    
    recommendations = similar_songs[similar_songs["name"] != song_name].sample(
        n=min(num_recommendations, len(similar_songs) - 1), random_state=42
    )
    return recommendations

# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="Spotify Recommendation System", layout="centered")

st.title("🎵 Spotify Song Recommendation System")

# Show dataset preview for debugging
if st.checkbox("Show dataset preview"):
    st.write(df.head())

# Song input
song_name = st.selectbox("Choose a song:", df["name"].unique())

# Number of recommendations
num_recs = st.slider("How many recommendations?", 1, 10, 5)

# Button
if st.button("Recommend Songs"):
    recommendations = recommend_songs(song_name, num_recs)
    
    if len(recommendations) == 0:
        st.error("❌ No recommendations found. Check if clustering worked.")
    else:
        st.subheader(f"Recommended songs for: {song_name}")
        for _, row in recommendations.iterrows():
            st.write(f"**{row['name']}**")
            
            # If dataset has Spotify URL column, show link
            if "url" in row or "spotify_url" in row or "track_url" in row:
                url = row.get("url") or row.get("spotify_url") or row.get("track_url")
                if pd.notna(url):
                    st.markdown(f"[▶️ Listen on Spotify]({url})")
            st.write("---")
