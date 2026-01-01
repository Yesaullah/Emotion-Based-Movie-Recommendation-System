import streamlit as st

# -------------------------------
# ML Imports
# -------------------------------
from classifier.model_io import load_or_train_model
from classifier.predictor import predict_sentiment

# KG Imports
from kg.kg_loader import load_kg
from kg.query import get_movies_by_emotion


# -------------------------------
# Streamlit Page Config
# -------------------------------
st.set_page_config(
    page_title="Emotion-Based Movie Recommendation System",
    layout="centered"
)

st.title("🎬 Emotion-Based Movie Recommendation System")

st.write(
    "This system reimplements the research paper by recommending movies "
    "based on emotion similarity between user input and movie reviews "
    "stored in a Knowledge Graph."
)

# -------------------------------
# Load Sentiment Model (Dual Mode)
# -------------------------------
@st.cache_resource
def load_model():
    return load_or_train_model()

vectorizer, model = load_model()

# -------------------------------
# Load Knowledge Graph
# -------------------------------
@st.cache_resource
def load_graph():
    return load_kg()

graph = load_graph()

# -------------------------------
# User Input
# -------------------------------
user_text = st.text_area(
    "Enter a sentence describing how you feel:",
    placeholder="e.g., This movie was absolutely amazing!",
    height=120
)

# -------------------------------
# Button Logic
# -------------------------------
if st.button("Get Recommendations"):
    if not user_text.strip():
        st.warning("Please enter some text.")
    else:
        # 1️⃣ Sentiment prediction
        sentiment = predict_sentiment(user_text, vectorizer, model)

        # 2️⃣ Sentiment → Emotion (paper logic)
        if sentiment == "positive":
            emotion = "joy"
        else:
            emotion = "sadness"

        # -------------------------------
        # Display NLP Results
        # -------------------------------
        st.subheader("Detected Sentiment")
        st.success(sentiment.capitalize())

        st.subheader("Inferred Emotion")
        st.info(emotion.capitalize())

        # -------------------------------
        # Knowledge Graph Recommendation
        # -------------------------------
        st.subheader("Recommended Movies")

        movies = get_movies_by_emotion(graph, emotion)

        if movies:
            for m in movies:
                st.write(f"- {m}")
        else:
            st.warning("No movies found with matching emotional reviews.")
