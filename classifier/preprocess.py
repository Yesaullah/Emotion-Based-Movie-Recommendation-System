import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Download required NLTK resources (runs once)
nltk.download("punkt")
nltk.download("stopwords")

STOP_WORDS = set(stopwords.words("english"))


def preprocess_text(text: str) -> str:
    """
    Clean and normalize input text for classification.
    Returns a single cleaned string.
    """

    # Lowercase
    text = text.lower()

    # Remove non-alphabetic characters
    text = re.sub(r"[^a-z\s]", "", text)

    # Tokenize
    tokens = word_tokenize(text)

    # Remove stopwords
    tokens = [word for word in tokens if word not in STOP_WORDS]

    # Join back into string (Naive Bayes-friendly)
    return " ".join(tokens)
