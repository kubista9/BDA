from gensim import corpora, models
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import nltk
nltk.download("punkt")
nltk.download("stopwords")

# Load and preprocess text
with open("my_corpus.txt") as f:
    text = f.read()

stop_words = set(stopwords.words("english"))
tokens = [word.lower() for word in word_tokenize(text) if word.isalpha() and word.lower() not in stop_words]
dictionary = corpora.Dictionary([tokens])
corpus = [dictionary.doc2bow(tokens)]

# Train LDA model
lda_model = models.LdaModel(corpus, num_topics=3, id2word=dictionary, passes=15)

# Print topics
for idx, topic in lda_model.print_topics():
    print(f"Topic {idx}: {topic}")
