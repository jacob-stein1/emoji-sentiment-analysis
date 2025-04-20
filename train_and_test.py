import sys
import os
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from text_embedders.tfidf import TFIDFTextEmbedder
from emoji_embedder.onehot import OneHotEmojiEmbedder
from utils import analyze_batch_pmf_differences

def train_and_analyze(dataset_path, test_texts=None):
    # Load dataset
    print("Loading dataset...")
    df = pd.read_csv(dataset_path)
    texts = df["text"]
    sentiments = df["sentiment"]
    
    # Initialize embedders
    print("Initializing embedders...")
    text_embedder = TFIDFTextEmbedder()
    emoji_embedder = OneHotEmojiEmbedder()
    
    # Train embedders
    print("Training embedders...")
    text_embedder.fit(texts)
    emoji_embedder.fit(texts)
    
    # Transform text and emoji embeddings
    print("Creating embeddings...")
    X_text = text_embedder.transform(texts).toarray()
    X_emoji = emoji_embedder.transform(texts)
    
    # Combine features
    X_combined = np.hstack((X_text, X_emoji))
    
    # Encode labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(sentiments)
    
    # Train models
    print("Training classifiers...")
    model_with_emoji = LogisticRegression(max_iter=1000, multi_class="multinomial", solver="lbfgs")
    model_with_emoji.fit(X_combined, y_encoded)
    
    model_without_emoji = LogisticRegression(max_iter=1000, multi_class="multinomial", solver="lbfgs")
    model_without_emoji.fit(X_text, y_encoded)
    
    # If test texts are provided, analyze them
    if test_texts:
        print("\nAnalyzing provided test texts...")
        results = analyze_batch_pmf_differences(
            texts=test_texts,
            classifier_with_emoji=model_with_emoji,
            classifier_without_emoji=model_without_emoji,
            text_embedder=text_embedder,
            emoji_embedder=emoji_embedder,
            label_encoder=label_encoder
        )
        
        print("\nAggregate Statistics:")
        for metric, value in results['aggregate_stats'].items():
            print(f"{metric}: {value:.4f}")
    
    return {
        'model_with_emoji': model_with_emoji,
        'model_without_emoji': model_without_emoji,
        'text_embedder': text_embedder,
        'emoji_embedder': emoji_embedder,
        'label_encoder': label_encoder
    }

if __name__ == "__main__":
    # Example usage
    dataset_path = "training.csv"
    test_texts = [
        "I love this! 😊",
        "This is terrible 😢",
        "It's okay 🙂",
        "The service was amazing! 🌟",
        "I'm so disappointed 😞"
    ]
    
    models = train_and_analyze(dataset_path, test_texts) 