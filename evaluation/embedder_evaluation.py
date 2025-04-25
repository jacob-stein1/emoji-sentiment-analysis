import sys
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Import all embedders
from text_embedders.tfidf import TFIDFTextEmbedder
from text_embedders.word2vec import Word2VecTextEmbedder
from text_embedders.fasttext import FastTextEmbedder
from text_embedders.bert import BERTEmbedder
from text_embedders.doc2vec import Doc2VecTextEmbedder

from emoji_embedder.tfidf import TFIDFEmojiEmbedder
from emoji_embedder.word2vec import Word2VecEmojiEmbedder
from emoji_embedder.fasttext import FastTextEmojiEmbedder
from emoji_embedder.bert import BERTEmojiEmbedder
from emoji_embedder.doc2vec import Doc2VecEmojiEmbedder

def evaluate_embedder_pair(text_embedder, emoji_embedder, X_train, X_test, y_train, y_test, label_encoder,
                            text_name, emoji_name, differing_samples):
    text_embedder.fit(X_train)
    emoji_embedder.fit(X_train)
    
    # Create embeddings
    X_train_text = text_embedder.transform(X_train)
    X_test_text = text_embedder.transform(X_test)
    X_train_emoji = emoji_embedder.transform(X_train)
    X_test_emoji = emoji_embedder.transform(X_test)
    
    if hasattr(X_train_text, 'toarray'):
        X_train_text = X_train_text.toarray()
        X_test_text = X_test_text.toarray()
    if hasattr(X_train_emoji, 'toarray'):
        X_train_emoji = X_train_emoji.toarray()
        X_test_emoji = X_test_emoji.toarray()
    
    X_train_combined = np.hstack((X_train_text, X_train_emoji))
    X_test_combined = np.hstack((X_test_text, X_test_emoji))
    
    model_text_only = LogisticRegression(max_iter=1000, multi_class="multinomial", solver="lbfgs")
    model_with_emoji = LogisticRegression(max_iter=1000, multi_class="multinomial", solver="lbfgs")
    
    model_text_only.fit(X_train_text, y_train)
    model_with_emoji.fit(X_train_combined, y_train)
    
    y_pred_text = model_text_only.predict(X_test_text)
    y_pred_combined = model_with_emoji.predict(X_test_combined)
    
    acc_text = accuracy_score(y_test, y_pred_text)
    acc_combined = accuracy_score(y_test, y_pred_combined)
    
    # Record differing predictions
    for i in range(len(X_test)):
        if y_pred_text[i] != y_pred_combined[i]:
            differing_samples.append({
                'text_embedder': text_name,
                'emoji_embedder': emoji_name,
                'text': X_test.iloc[i],
                'text_only_prediction': label_encoder.inverse_transform([y_pred_text[i]])[0],
                'hybrid_prediction': label_encoder.inverse_transform([y_pred_combined[i]])[0],
                'true_class': label_encoder.inverse_transform([y_test[i]])[0]
            })
    
    return acc_combined - acc_text

def main():
    print("Loading dataset...")
    df = pd.read_csv("training.csv")
    
    X_train, X_test, y_train, y_test = train_test_split(
        df["text"], df["sentiment"], 
        test_size=0.2, 
        random_state=42
    )
    
    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train)
    y_test_encoded = label_encoder.transform(y_test)
    
    text_embedders = {
        'TF-IDF': TFIDFTextEmbedder(),
        'Word2Vec': Word2VecTextEmbedder(),
        'FastText': FastTextEmbedder(),
        'BERT': BERTEmbedder(),
        'Doc2Vec': Doc2VecTextEmbedder()
    }
    
    emoji_embedders = {
        'TF-IDF': TFIDFEmojiEmbedder(),
        'Word2Vec': Word2VecEmojiEmbedder(),
        'FastText': FastTextEmojiEmbedder(),
        'BERT': BERTEmojiEmbedder(),
        'Doc2Vec': Doc2VecEmojiEmbedder()
    }
    
    results = []
    differing_samples = []
    
    total_combinations = len(text_embedders) * len(emoji_embedders)
    current = 0
    
    print("\nEvaluating embedder combinations...")
    for text_name, text_embedder in text_embedders.items():
        for emoji_name, emoji_embedder in emoji_embedders.items():
            current += 1
            print(f"\nProgress: {current}/{total_combinations}")
            print(f"Testing {text_name} + {emoji_name}")
            
            diff = evaluate_embedder_pair(
                text_embedder, 
                emoji_embedder,
                X_train, X_test,
                y_train_encoded, y_test_encoded,
                label_encoder,
                text_name,
                emoji_name,
                differing_samples
            )
            results.append({
                'text_embedder': text_name,
                'emoji_embedder': emoji_name,
                'accuracy_difference': diff
            })
            print(f"Accuracy difference: {diff:.4f}")
    
    results_df = pd.DataFrame(results)
    
    pivot_table = results_df.pivot(
        index='text_embedder',
        columns='emoji_embedder',
        values='accuracy_difference'
    )
    
    print("\nResults:")
    print(pivot_table)
    
    results_df.to_csv('embedder_evaluation_results.csv', index=False)
    print("\nResults saved to 'embedder_evaluation_results.csv'")
    
    # Save differing samples
    if differing_samples:
        differing_df = pd.DataFrame(differing_samples)
        differing_df.to_csv('differing_predictions_all.csv', index=False)
        print(f"Saved {len(differing_samples)} differing predictions to 'differing_predictions_all.csv'")
    
    best_result = results_df.loc[results_df['accuracy_difference'].idxmax()]
    print(f"\nBest combination:")
    print(f"Text embedder: {best_result['text_embedder']}")
    print(f"Emoji embedder: {best_result['emoji_embedder']}")
    print(f"Accuracy difference: {best_result['accuracy_difference']:.4f}")

if __name__ == "__main__":
    main() 