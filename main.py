import re
from collections import Counter
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from textblob import TextBlob

class BasicTextAnalyzer:
    def __init__(self):
        self.text_data = ""
        self.words = []
        self.word_counts = None
    
    def load_text(self, text_input):
        """Load text from string or file"""
        if isinstance(text_input, str):
            if text_input.endswith('.txt'):
                with open(text_input, 'r', encoding='utf-8') as file:
                    self.text_data = file.read()
            else:
                self.text_data = text_input
        else:
            raise ValueError("Input must be string or file path")
        
        # Basic word tokenization (no NLTK needed)
        self.words = [word.lower() for word in re.findall(r'\w+', self.text_data)]
        self.word_counts = Counter(self.words)
    
    def basic_stats(self):
        """Return basic text statistics"""
        if not self.text_data:
            return {}
        
        return {
            'total_chars': len(self.text_data),
            'total_words': len(self.words),
            'unique_words': len(set(self.words)),
            'lexical_diversity': len(set(self.words)) / len(self.words) if self.words else 0,
            'top_words': self.word_counts.most_common(10)
        }
    
    def sentiment_analysis(self):
        """Basic sentiment analysis using TextBlob"""
        blob = TextBlob(self.text_data)
        return {
            'polarity': blob.sentiment.polarity,
            'subjectivity': blob.sentiment.subjectivity,
            'assessment': "Positive" if blob.sentiment.polarity > 0.1 else 
                         "Negative" if blob.sentiment.polarity < -0.1 else 
                         "Neutral"
        }
    
    def create_wordcloud(self):
        """Generate and display a word cloud"""
        wordcloud = WordCloud(width=800, height=400, 
                             background_color='white').generate(self.text_data)
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.show()
    
    def plot_word_frequency(self, top_n=15):
        """Plot top words frequency"""
        top_words = self.word_counts.most_common(top_n)
        words, counts = zip(*top_words)
        
        plt.figure(figsize=(10, 5))
        plt.barh(words, counts, color='skyblue')
        plt.title(f'Top {top_n} Most Frequent Words')
        plt.xlabel('Frequency')
        plt.tight_layout()
        plt.show()

def demo():
    """Demo the basic analyzer"""
    sample_text = """
    Artificial intelligence is transforming our world. Machine learning helps computers 
    learn from data. Many companies use AI for business applications. 
    The future of AI looks promising but we must consider ethical implications.
    """
    
    print("🚀 Basic Text Analysis Demo")
    print("=" * 40)
    
    analyzer = BasicTextAnalyzer()
    analyzer.load_text(sample_text)
    
    # Show stats
    stats = analyzer.basic_stats()
    print("\n📊 Basic Statistics:")
    print(f"Total characters: {stats['total_chars']}")
    print(f"Total words: {stats['total_words']}")
    print(f"Unique words: {stats['unique_words']}")
    print(f"Top words: {[w[0] for w in stats['top_words']]}")
    
    # Show sentiment
    sentiment = analyzer.sentiment_analysis()
    print("\n😊 Sentiment Analysis:")
    print(f"Polarity: {sentiment['polarity']:.2f} ({sentiment['assessment']})")
    print(f"Subjectivity: {sentiment['subjectivity']:.2f}")
    
    # Create visualizations
    print("\n🖼️ Generating visualizations...")
    analyzer.create_wordcloud()
    analyzer.plot_word_frequency()

if __name__ == "__main__":
    demo()