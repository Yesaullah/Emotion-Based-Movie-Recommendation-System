from pathlib import Path
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from classifier.preprocess import preprocess_text


def train_model(limit=500):
    texts = []
    labels = []

    base_path = Path("data/imdb")

    for label, folder in [("positive", "pos"), ("negative", "neg")]:
        folder_path = base_path / folder

        for i, file in enumerate(folder_path.iterdir()):
            if i >= limit:
                break
            if file.suffix == ".txt":
                text = file.read_text(encoding="utf-8", errors="ignore")
                texts.append(preprocess_text(text))
                labels.append(label)

    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(texts)

    model = MultinomialNB()
    model.fit(X, labels)

    return vectorizer, model
