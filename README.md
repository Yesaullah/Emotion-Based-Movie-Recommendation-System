# Emotion-Based Movie Recommendation System  
**(Research Paper Reimplementation)**

This project reimplements a research paper that recommends movies based on **emotional similarity** between **user input text** and **movie reviews** stored in a **Knowledge Graph**.

The system combines **Machine Learning (Naive Bayes)** with a **Knowledge Graph (RDF + ONYX ontology)** and provides an interactive interface using **Streamlit**.

---

## Project Overview

### Workflow

```
User Input Text
   ↓
Naive Bayes Sentiment Analysis (IMDb Reviews)
   ↓
Emotion Inference (e.g., joy, sadness)
   ↓
SPARQL Query on Knowledge Graph (ONYX)
   ↓
Emotion-Based Movie Recommendations
```

---

## Project Structure

```
movie-emotion-kg/
│
├── app.py
├── requirements.txt
│
├── classifier/
│   ├── preprocess.py
│   ├── model.py
│   ├── predictor.py
│   └── model_io.py
│
├── kg/
│   ├── kg_loader.py
│   └── query.py
│
├── scripts/
│   └── train_and_save.py
│
├── data/
│   └── imdb/
│       ├── pos/
│       └── neg/
│
├── models/
│   ├── vectorizer.pkl
│   └── sentiment_model.pkl
│
└── output/
    └── movie_emotion_kg.ttl
```

---

## Installation & Setup

### Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Running the Project

### Optional: Pre-train the Model

```bash
python -m scripts.train_and_save
```

### Run the Application

```bash
streamlit run app.py
```

Open browser at: http://localhost:8501

---

## Example Input

```
This movie was amazing and inspiring
```

---

## 🎓 Notes

- Uses IMDb reviews
- Uses Naive Bayes
- Uses ONYX-based Knowledge Graph
- Fully paper-faithful implementation

---
