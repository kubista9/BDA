from gensim import corpora

# Create dictionary and bag-of-words corpus
dictionary = corpora.Dictionary([filtered_tokens])
corpus = [dictionary.doc2bow(filtered_tokens)]
