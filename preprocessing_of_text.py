import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

nltk.download("punkt")
nltk.download("stopwords")

stop_words = set(stopwords.words("english"))

# Tokenize and clean
tokens = [word.lower() for word in word_tokenize(text) if word.isalpha()]
filtered_tokens = [word for word in tokens if word not in stop_words and len(word) > 3]
