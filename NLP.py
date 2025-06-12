import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# NLP Libraries
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.chunk import ne_chunk
from nltk.tag import pos_tag
import spacy
from textblob import TextBlob
from collections import Counter, defaultdict
import re

# Topic Modeling
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation, NMF
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity

# Sentiment Analysis
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
    nltk.data.find('corpora/wordnet')
    nltk.data.find('taggers/averaged_perceptron_tagger')
    nltk.data.find('chunkers/maxent_ne_chunker')
    nltk.data.find('corpora/words')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')
    nltk.download('averaged_perceptron_tagger')
    nltk.download('maxent_ne_chunker')
    nltk.download('words')

class AdvancedNLPAnalyzer:
    def __init__(self, language='english'):
        """
        Initialize the NLP Analyzer with required components
        """
        self.language = language
        self.stop_words = set(stopwords.words(language))
        self.lemmatizer = WordNetLemmatizer()
        self.sentiment_analyzer = SentimentIntensityAnalyzer()
        
        # Try to load SpaCy model
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            print("SpaCy model not found. Install with: python -m spacy download en_core_web_sm")
            self.nlp = None
        
        # Initialize storage for analysis results
        self.text_data = None
        self.processed_text = None
        self.sentences = None
        self.topics = None
        self.sentiment_scores = None
        
    def load_text(self, text_input):
        """
        Load text from string, file, or list of texts
        """
        if isinstance(text_input, str):
            if text_input.endswith('.txt'):
                with open(text_input, 'r', encoding='utf-8') as file:
                    self.text_data = file.read()
            else:
                self.text_data = text_input
        elif isinstance(text_input, list):
            self.text_data = ' '.join(text_input)
        else:
            raise ValueError("Input must be string, file path, or list of strings")
        
        self.sentences = sent_tokenize(self.text_data)
        print(f"Loaded text with {len(self.text_data)} characters and {len(self.sentences)} sentences")
        
    def preprocess_text(self, remove_punct=True, lowercase=True, remove_stopwords=True, lemmatize=True):
        """
        Comprehensive text preprocessing
        """
        text = self.text_data
        
        if lowercase:
            text = text.lower()
        
        if remove_punct:
            text = re.sub(r'[^\w\s]', ' ', text)
            text = re.sub(r'\s+', ' ', text)
        
        # Tokenize
        tokens = word_tokenize(text)
        
        if remove_stopwords:
            tokens = [token for token in tokens if token not in self.stop_words]
        
        if lemmatize:
            tokens = [self.lemmatizer.lemmatize(token) for token in tokens]
        
        # Remove short words and numbers
        tokens = [token for token in tokens if len(token) > 2 and not token.isdigit()]
        
        self.processed_text = ' '.join(tokens)
        return self.processed_text
    
    def basic_statistics(self):
        """
        Generate basic text statistics
        """
        if not self.text_data:
            raise ValueError("No text loaded")
        
        words = word_tokenize(self.text_data.lower())
        words = [word for word in words if word.isalpha()]
        
        stats = {
            'total_characters': len(self.text_data),
            'total_words': len(words),
            'total_sentences': len(self.sentences),
            'avg_words_per_sentence': len(words) / len(self.sentences),
            'unique_words': len(set(words)),
            'lexical_diversity': len(set(words)) / len(words),
            'most_common_words': Counter(words).most_common(10)
        }
        
        return stats
    
    def sentiment_analysis(self):
        """
        Perform sentiment analysis on sentences and overall text
        """
        if not self.sentences:
            raise ValueError("No text loaded")
        
        # Sentence-level sentiment
        sentence_sentiments = []
        for sentence in self.sentences:
            # VADER sentiment
            vader_scores = self.sentiment_analyzer.polarity_scores(sentence)
            
            # TextBlob sentiment
            blob = TextBlob(sentence)
            textblob_sentiment = blob.sentiment
            
            sentence_sentiments.append({
                'sentence': sentence,
                'vader_compound': vader_scores['compound'],
                'vader_positive': vader_scores['pos'],
                'vader_negative': vader_scores['neg'],
                'vader_neutral': vader_scores['neu'],
                'textblob_polarity': textblob_sentiment.polarity,
                'textblob_subjectivity': textblob_sentiment.subjectivity
            })
        
        # Overall sentiment
        overall_vader = self.sentiment_analyzer.polarity_scores(self.text_data)
        overall_textblob = TextBlob(self.text_data).sentiment
        
        self.sentiment_scores = {
            'sentence_level': sentence_sentiments,
            'overall_vader': overall_vader,
            'overall_textblob': overall_textblob
        }
        
        return self.sentiment_scores
    
    def topic_modeling(self, n_topics=5, method='lda', n_words=10):
        """
        Perform topic modeling using LDA or NMF
        """
        if not self.processed_text:
            self.preprocess_text()
        
        # Prepare documents (sentences)
        processed_sentences = []
        for sentence in self.sentences:
            processed_sentence = self._preprocess_sentence(sentence)
            if processed_sentence:
                processed_sentences.append(processed_sentence)
        
        if len(processed_sentences) < n_topics:
            n_topics = max(1, len(processed_sentences) // 2)
        
        # Vectorization
        if method.lower() == 'lda':
            vectorizer = CountVectorizer(max_features=1000, min_df=2, max_df=0.8)
            doc_term_matrix = vectorizer.fit_transform(processed_sentences)
            
            # LDA
            lda = LatentDirichletAllocation(n_components=n_topics, random_state=42, max_iter=100)
            lda.fit(doc_term_matrix)
            
            model = lda
        else:  # NMF
            vectorizer = TfidfVectorizer(max_features=1000, min_df=2, max_df=0.8)
            doc_term_matrix = vectorizer.fit_transform(processed_sentences)
            
            # NMF
            nmf = NMF(n_components=n_topics, random_state=42, max_iter=100)
            nmf.fit(doc_term_matrix)
            
            model = nmf
        
        # Extract topics
        feature_names = vectorizer.get_feature_names_out()
        topics = []
        
        for topic_idx, topic in enumerate(model.components_):
            top_words_idx = topic.argsort()[-n_words:][::-1]
            top_words = [feature_names[i] for i in top_words_idx]
            topic_weights = [topic[i] for i in top_words_idx]
            
            topics.append({
                'topic_id': topic_idx,
                'words': top_words,
                'weights': topic_weights
            })
        
        # Document-topic distribution
        doc_topic_dist = model.transform(doc_term_matrix)
        
        self.topics = {
            'topics': topics,
            'doc_topic_distribution': doc_topic_dist,
            'method': method,
            'vectorizer': vectorizer,
            'model': model
        }
        
        return self.topics
    
    def _preprocess_sentence(self, sentence):
        """Helper method to preprocess individual sentences"""
        sentence = sentence.lower()
        sentence = re.sub(r'[^\w\s]', ' ', sentence)
        tokens = word_tokenize(sentence)
        tokens = [self.lemmatizer.lemmatize(token) for token in tokens 
                 if token not in self.stop_words and len(token) > 2 and not token.isdigit()]
        return ' '.join(tokens)
    
    def named_entity_recognition(self):
        """
        Extract named entities using NLTK and SpaCy
        """
        entities = {'nltk': [], 'spacy': []}
        
        # NLTK NER
        tokens = word_tokenize(self.text_data)
        pos_tags = pos_tag(tokens)
        tree = ne_chunk(pos_tags)
        
        for chunk in tree:
            if hasattr(chunk, 'label'):
                entity = ' '.join([token for token, pos in chunk.leaves()])
                entities['nltk'].append({'entity': entity, 'label': chunk.label()})
        
        # SpaCy NER (if available)
        if self.nlp:
            doc = self.nlp(self.text_data)
            for ent in doc.ents:
                entities['spacy'].append({'entity': ent.text, 'label': ent.label_})
        
        return entities
    
    def create_visualizations(self, save_plots=False, output_dir='./'):
        """
        Create comprehensive visualizations
        """
        if not self.sentiment_scores or not self.topics:
            print("Running sentiment analysis and topic modeling first...")
            self.sentiment_analysis()
            self.topic_modeling()
        
        # Set up the plotting style
        plt.style.use('seaborn-v0_8')
        fig = plt.figure(figsize=(20, 15))
        
        # 1. Word Cloud
        plt.subplot(3, 3, 1)
        wordcloud = WordCloud(width=400, height=300, background_color='white').generate(self.processed_text)
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.title('Word Cloud', fontsize=14, fontweight='bold')
        plt.axis('off')
        
        # 2. Sentiment Distribution
        plt.subplot(3, 3, 2)
        sentiments = [s['vader_compound'] for s in self.sentiment_scores['sentence_level']]
        plt.hist(sentiments, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        plt.title('Sentiment Distribution (VADER)', fontsize=14, fontweight='bold')
        plt.xlabel('Sentiment Score')
        plt.ylabel('Frequency')
        
        # 3. Sentiment Over Time (sentences)
        plt.subplot(3, 3, 3)
        sentence_nums = range(len(sentiments))
        plt.plot(sentence_nums, sentiments, alpha=0.7, color='purple')
        plt.title('Sentiment Trend Across Text', fontsize=14, fontweight='bold')
        plt.xlabel('Sentence Number')
        plt.ylabel('Sentiment Score')
        
        # 4. Top Words Frequency
        plt.subplot(3, 3, 4)
        stats = self.basic_statistics()
        words, counts = zip(*stats['most_common_words'])
        plt.barh(words, counts, color='lightcoral')
        plt.title('Top 10 Most Common Words', fontsize=14, fontweight='bold')
        plt.xlabel('Frequency')
        
        # 5. Topic Modeling Results
        plt.subplot(3, 3, 5)
        if self.topics:
            topic_labels = [f"Topic {i+1}" for i in range(len(self.topics['topics']))]
            topic_sizes = [sum(topic['weights']) for topic in self.topics['topics']]
            plt.pie(topic_sizes, labels=topic_labels, autopct='%1.1f%%', startangle=90)
            plt.title('Topic Distribution', fontsize=14, fontweight='bold')
        
        # 6. Sentence Length Distribution
        plt.subplot(3, 3, 6)
        sentence_lengths = [len(word_tokenize(sentence)) for sentence in self.sentences]
        plt.hist(sentence_lengths, bins=20, alpha=0.7, color='lightgreen', edgecolor='black')
        plt.title('Sentence Length Distribution', fontsize=14, fontweight='bold')
        plt.xlabel('Number of Words')
        plt.ylabel('Frequency')
        
        # 7. Polarity vs Subjectivity
        plt.subplot(3, 3, 7)
        polarities = [s['textblob_polarity'] for s in self.sentiment_scores['sentence_level']]
        subjectivities = [s['textblob_subjectivity'] for s in self.sentiment_scores['sentence_level']]
        plt.scatter(polarities, subjectivities, alpha=0.6, color='orange')
        plt.title('Polarity vs Subjectivity', fontsize=14, fontweight='bold')
        plt.xlabel('Polarity')
        plt.ylabel('Subjectivity')
        
        # 8. Topic Words Heatmap
        plt.subplot(3, 3, 8)
        if self.topics and len(self.topics['topics']) > 1:
            topic_word_matrix = []
            all_words = set()
            for topic in self.topics['topics']:
                all_words.update(topic['words'][:5])
            
            all_words = list(all_words)
            for topic in self.topics['topics']:
                topic_scores = []
                for word in all_words:
                    if word in topic['words']:
                        idx = topic['words'].index(word)
                        topic_scores.append(topic['weights'][idx])
                    else:
                        topic_scores.append(0)
                topic_word_matrix.append(topic_scores)
            
            sns.heatmap(topic_word_matrix, xticklabels=all_words, 
                       yticklabels=[f'Topic {i+1}' for i in range(len(topic_word_matrix))],
                       cmap='YlOrRd', annot=False)
            plt.title('Topic-Word Intensity', fontsize=14, fontweight='bold')
            plt.xticks(rotation=45)
        
        # 9. Text Statistics Summary
        plt.subplot(3, 3, 9)
        plt.axis('off')
        stats_text = f"""
        Text Statistics:
        
        Total Characters: {stats['total_characters']:,}
        Total Words: {stats['total_words']:,}
        Total Sentences: {stats['total_sentences']:,}
        Unique Words: {stats['unique_words']:,}
        Lexical Diversity: {stats['lexical_diversity']:.3f}
        Avg Words/Sentence: {stats['avg_words_per_sentence']:.1f}
        
        Overall Sentiment (VADER): {self.sentiment_scores['overall_vader']['compound']:.3f}
        Overall Polarity (TextBlob): {self.sentiment_scores['overall_textblob'].polarity:.3f}
        """
        plt.text(0.1, 0.9, stats_text, fontsize=11, verticalalignment='top', 
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        plt.title('Summary Statistics', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        
        if save_plots:
            plt.savefig(f'{output_dir}nlp_analysis_dashboard.png', dpi=300, bbox_inches='tight')
        
        plt.show()
        
        # Create interactive topic visualization
        self._create_interactive_topic_viz(save_plots, output_dir)
    
    def _create_interactive_topic_viz(self, save_plots=False, output_dir='./'):
        """Create interactive topic modeling visualization"""
        if not self.topics:
            return
        
        # Prepare data for visualization
        topics_data = []
        for i, topic in enumerate(self.topics['topics']):
            for j, (word, weight) in enumerate(zip(topic['words'][:10], topic['weights'][:10])):
                topics_data.append({
                    'Topic': f'Topic {i+1}',
                    'Word': word,
                    'Weight': weight,
                    'Rank': j+1
                })
        
        df_topics = pd.DataFrame(topics_data)
        
        # Create interactive plot
        fig = px.bar(df_topics, x='Weight', y='Word', color='Topic',
                    facet_col='Topic', facet_col_wrap=3,
                    title='Interactive Topic Modeling Results',
                    height=600)
        
        fig.update_layout(showlegend=False)
        
        if save_plots:
            fig.write_html(f'{output_dir}interactive_topics.html')
        
        fig.show()
    
    def generate_report(self, output_file='nlp_analysis_report.txt'):
        """
        Generate a comprehensive text report
        """
        if not self.sentiment_scores or not self.topics:
            self.sentiment_analysis()
            self.topic_modeling()
        
        stats = self.basic_statistics()
        entities = self.named_entity_recognition()
        
        report = f"""
NLP ANALYSIS REPORT
==================

TEXT OVERVIEW:
--------------
• Total Characters: {stats['total_characters']:,}
• Total Words: {stats['total_words']:,}
• Total Sentences: {stats['total_sentences']:,}
• Unique Words: {stats['unique_words']:,}
• Lexical Diversity: {stats['lexical_diversity']:.3f}
• Average Words per Sentence: {stats['avg_words_per_sentence']:.1f}

SENTIMENT ANALYSIS:
-------------------
Overall Sentiment (VADER Compound): {self.sentiment_scores['overall_vader']['compound']:.3f}
Overall Polarity (TextBlob): {self.sentiment_scores['overall_textblob'].polarity:.3f}
Overall Subjectivity (TextBlob): {self.sentiment_scores['overall_textblob'].subjectivity:.3f}

Sentiment Interpretation:
• VADER Score: {"Positive" if self.sentiment_scores['overall_vader']['compound'] > 0.05 else "Negative" if self.sentiment_scores['overall_vader']['compound'] < -0.05 else "Neutral"}
• TextBlob Polarity: {"Positive" if self.sentiment_scores['overall_textblob'].polarity > 0.1 else "Negative" if self.sentiment_scores['overall_textblob'].polarity < -0.1 else "Neutral"}

TOPIC MODELING RESULTS:
-----------------------
Number of Topics Identified: {len(self.topics['topics'])}
Method Used: {self.topics['method'].upper()}

"""
        
        for i, topic in enumerate(self.topics['topics']):
            report += f"\nTopic {i+1}:\n"
            report += f"Key Words: {', '.join(topic['words'][:8])}\n"
        
        report += f"""
MOST COMMON WORDS:
------------------
"""
        for word, count in stats['most_common_words']:
            report += f"• {word}: {count}\n"
        
        report += f"""
NAMED ENTITIES (Sample):
------------------------
"""
        if entities['spacy']:
            unique_entities = list({ent['entity']: ent for ent in entities['spacy']}.values())[:10]
            for ent in unique_entities:
                report += f"• {ent['entity']} ({ent['label']})\n"
        elif entities['nltk']:
            unique_entities = list({ent['entity']: ent for ent in entities['nltk']}.values())[:10]
            for ent in unique_entities:
                report += f"• {ent['entity']} ({ent['label']})\n"
        
        # Save report
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(report)
        
        print(f"Report saved to {output_file}")
        return report

# Example usage and demonstration
def demo_analysis():
    """
    Demonstration of the NLP analyzer with sample text
    """
    # Sample text for demonstration
    sample_text = """
    Artificial intelligence is transforming the world in unprecedented ways. Machine learning algorithms 
    are being deployed across various industries, from healthcare to finance, revolutionizing how we 
    approach complex problems. The development of natural language processing has enabled computers 
    to understand and generate human language with remarkable accuracy.
    
    However, the rapid advancement of AI technology also raises important ethical concerns. Questions 
    about privacy, bias, and the future of work need careful consideration. Many researchers and 
    policymakers are working together to ensure that AI development remains beneficial for society.
    
    The future of artificial intelligence looks incredibly promising. New breakthroughs in deep learning 
    and neural networks continue to push the boundaries of what's possible. From autonomous vehicles 
    to personalized medicine, AI applications are becoming more sophisticated and widespread.
    
    Education and public awareness about AI are crucial for its successful integration into society. 
    As we move forward, collaboration between technologists, ethicists, and policymakers will be 
    essential to harness the full potential of artificial intelligence while mitigating its risks.
    """
    
    print("Starting NLP Analysis Demo...")
    print("=" * 50)
    
    # Initialize analyzer
    analyzer = AdvancedNLPAnalyzer()
    
    # Load text
    analyzer.load_text(sample_text)
    
    # Preprocess
    analyzer.preprocess_text()
    
    # Run analyses
    print("Running basic statistics...")
    stats = analyzer.basic_statistics()
    
    print("Running sentiment analysis...")
    sentiment = analyzer.sentiment_analysis()
    
    print("Running topic modeling...")
    topics = analyzer.topic_modeling(n_topics=3)
    
    print("Generating visualizations...")
    analyzer.create_visualizations()
    
    print("Generating report...")
    report = analyzer.generate_report()
    
    print("\nDemo completed! Check the generated files and visualizations.")
    
    return analyzer

if __name__ == "__main__":
    # Run demonstration
    analyzer = demo_analysis()