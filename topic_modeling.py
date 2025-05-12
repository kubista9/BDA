import os
import nltk
from gensim import corpora
from gensim.models import LdaModel
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# === SETUP ===
nltk.download("punkt")
nltk.download("stopwords")

INPUT_FILE = "data/my_corpus.txt"
NUM_TOPICS = 3
OUTPUT_FILE = "output/topics.txt"

# === LOAD & PREPROCESS TEXT ===
with open(INPUT_FILE, "r", encoding="utf-8") as file:
    text = file.read()

tokens = [word.lower() for word in word_tokenize(text) if word.isalpha()]
stop_words = set(stopwords.words("english"))
filtered_tokens = [word for word in tokens if word not in stop_words and len(word) > 3]

# === GENSIM DICTIONARY AND CORPUS ===
dictionary = corpora.Dictionary([filtered_tokens])
corpus = [dictionary.doc2bow(filtered_tokens)]

# === TRAIN LDA MODEL ===
lda_model = LdaModel(corpus=corpus, id2word=dictionary, num_topics=NUM_TOPICS, passes=15)

# === OUTPUT TOPICS ===
topics = lda_model.print_topics(num_words=5)
os.makedirs("output", exist_ok=True)

with open(OUTPUT_FILE, "w") as f:
    for i, topic in topics:
        line = f"Topic {i}: {topic}"
        print(line)
        f.write(line + "\n")
