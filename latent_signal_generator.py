import torch
from sentence_transformers import SentenceTransformer, util

class LatentSpaceIndicator:
    def __init__(self):
        print("Loading Latent Space Model... (FinBERT removed for unified architecture)")
        # We only need one model now!
        self.embedder = SentenceTransformer('all-mpnet-base-v2')
        
        self.concept_centroids = {}
        self.sentiment_centroids = {}
        
        print("Model loaded successfully!")

    def define_concept(self, concept_name, key_phrases):
        """Creates the centroid for an industry or topic."""
        vectors = self.embedder.encode(key_phrases, convert_to_tensor=True)
        self.concept_centroids[concept_name] = torch.mean(vectors, dim=0)
        print(f"Concept '{concept_name}' defined.")

    def define_sentiment(self, sentiment_name, key_phrases):
        """Creates the centroid for an emotional or financial tone."""
        vectors = self.embedder.encode(key_phrases, convert_to_tensor=True)
        self.sentiment_centroids[sentiment_name] = torch.mean(vectors, dim=0)
        print(f"Sentiment '{sentiment_name}' defined.")

    def analyze_news(self, news_text):
        """Projects the news into both concept and sentiment spaces."""
        news_vector = self.embedder.encode(news_text, convert_to_tensor=True)
        
        # 1. Concept Relevance (How much is it about AI or Agriculture?)
        relevance_scores = {}
        for concept, centroid in self.concept_centroids.items():
            relevance_scores[concept] = max(0, util.cos_sim(news_vector, centroid).item())
            
        # 2. Sentiment Proximity (Is it closer to Euphoria, Panic, or Neutral?)
        sentiment_scores = {}
        for sentiment, centroid in self.sentiment_centroids.items():
            sentiment_scores[sentiment] = max(0, util.cos_sim(news_vector, centroid).item())
            
        # 3. Calculate a Net Directional Signal 
        # (Assuming you defined 'Positive' and 'Negative' sentiments)
        net_sentiment_multiplier = 0
        if "Positive" in sentiment_scores and "Negative" in sentiment_scores:
            # We subtract the negative proximity from the positive proximity
            # This creates a dynamic range between roughly -1.0 and +1.0
            net_sentiment_multiplier = sentiment_scores["Positive"] - sentiment_scores["Negative"]
            
        directed_signals = {}
        for concept, relevance in relevance_scores.items():
            directed_signals[concept] = relevance * net_sentiment_multiplier

        return {
            "text": news_text,
            "relevance_scores": relevance_scores,
            "sentiment_scores": sentiment_scores,
            "net_sentiment": net_sentiment_multiplier,
            "directed_signals": directed_signals
        }

# ==========================================
# USAGE EXAMPLE
# ==========================================
if __name__ == "__main__":
    indicator = LatentSpaceIndicator()

    # --- 1. Define Concepts ---
    indicator.define_concept("Artificial_Intelligence", [
        "artificial intelligence breakthrough", 
        "machine learning algorithm", 
        "generative AI models"
    ])
    indicator.define_concept("Agriculture", [
        "farming equipment", 
        "crop yield harvest", 
        "agricultural technology"
    ])

    # --- 2. Define Custom Sentiments ---
    # Using highly polarized, descriptive sentences to avoid the antonym trap
    indicator.define_sentiment("Positive", [
        "The company reported record-breaking revenue and massive profits.",
        "Unprecedented demand has led to a highly successful quarter.",
        "The financial outlook is extremely optimistic and bullish."
    ])
    
    indicator.define_sentiment("Negative", [
        "The company suffered heavy financial losses and declining revenue.",
        "A devastating crisis has caused bankruptcy and massive layoffs.",
        "The market is crashing, leading to a highly pessimistic outlook."
    ])
    
    indicator.define_sentiment("Uncertain", [
        "The outcome remains unclear amid volatile market conditions.",
        "Investors are awaiting further regulatory guidance before acting.",
        "It is too early to tell what the long-term impact will be."
    ])

    # --- 3. Test the Engine ---
    news_headline = "Nvidia announces massive profits due to unprecedented demand for generative AI training chips."
    print(f"\nAnalyzing news: '{news_headline}'\n")
    
    results = indicator.analyze_news(news_headline)
    
    print("--- Sentiment Profile ---")
    for sentiment, score in results['sentiment_scores'].items():
        print(f"- {sentiment}: {score:.4f}")
    
    print(f"\n=> Net Sentiment Multiplier (Pos - Neg): {results['net_sentiment']:.4f}")

    print("\n--- Directed Signals ---")
    for concept, signal in results['directed_signals'].items():
        print(f"- {concept}: {signal:.4f}")