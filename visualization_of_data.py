import matplotlib.pyplot as plt
from nltk import FreqDist

# Assume `tokens` from before
freq_dist = FreqDist(tokens)
most_common = freq_dist.most_common(10)

# Plot
words, counts = zip(*most_common)
plt.bar(words, counts)
plt.xticks(rotation=45)
plt.title("Top 10 Most Common Words")
plt.show()