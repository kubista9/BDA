topics = lda_model.print_topics(num_words=5)
for i, topic in topics:
    print(f"Topic {i}: {topic}")
