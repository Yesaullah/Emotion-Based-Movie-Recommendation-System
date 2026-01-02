# Emotion-Based Movie Recommendation System
### Research Paper Reimplementation using Machine Learning and Knowledge Graphs

---

## Project Overview

This project is a faithful reimplementation of a research paper that proposes an emotion-based movie recommendation system.
Unlike traditional recommender systems that rely on genres or ratings, this system recommends movies by matching the emotion
expressed in user input text with emotions expressed in movie reviews stored in a Knowledge Graph.

The project integrates:
- Machine Learning (Naive Bayes)
- Natural Language Processing
- Knowledge Graphs (RDF)
- ONYX Emotion Ontology
- SPARQL-based semantic querying
- Streamlit for deployment

---

## Workflow

User Input Text  
→ Text Preprocessing  
→ Naive Bayes Sentiment Classification  
→ Emotion Inference (Joy / Sadness)  
→ SPARQL Query on Knowledge Graph  
→ Emotion-Based Movie Recommendations  

---

## Dataset Used

IMDb Movie Review Dataset:
- Positive and negative labeled reviews
- Used only during training
- Stored in data/imdb/pos and data/imdb/neg

---

## Sentiment & Emotion Modeling

- Algorithm: Multinomial Naive Bayes
- Feature extraction: CountVectorizer (Bag-of-Words)
- Emotion mapping:
  - Positive → Joy
  - Negative → Sadness

---

## Knowledge Graph

- Represents movies, reviews, and emotions
- Uses schema.org and ONYX ontology
- Stored as output/movie_emotion_kg.ttl
- Queried using SPARQL using rdflib

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

## How to Run

pip install -r requirements.txt  
python -m scripts.train_and_save  
streamlit run app.py  

---
