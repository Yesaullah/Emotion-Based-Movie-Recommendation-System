import pickle
from pathlib import Path
from classifier.model import train_model

MODEL_DIR = Path("models")
VECTORIZER_PATH = MODEL_DIR / "vectorizer.pkl"
MODEL_PATH = MODEL_DIR / "sentiment_model.pkl"


def load_or_train_model():
    if VECTORIZER_PATH.exists() and MODEL_PATH.exists():
        with open(VECTORIZER_PATH, "rb") as f:
            vectorizer = pickle.load(f)
        with open(MODEL_PATH, "rb") as f:
            model = pickle.load(f)
        return vectorizer, model

    MODEL_DIR.mkdir(exist_ok=True)
    vectorizer, model = train_model()

    with open(VECTORIZER_PATH, "wb") as f:
        pickle.dump(vectorizer, f)
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)

    return vectorizer, model
