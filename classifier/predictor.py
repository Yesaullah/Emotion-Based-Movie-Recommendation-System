from classifier.preprocess import preprocess_text


def predict_sentiment(text: str, vectorizer, model) -> str:
    clean_text = preprocess_text(text)
    X = vectorizer.transform([clean_text])
    return model.predict(X)[0]
