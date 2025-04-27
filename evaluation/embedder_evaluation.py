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

def evaluate_embedder_pair(text_embedder, 
                            emoji_embedder, 
                            X_train, X_test, 
                            y_train, y_test, 
                            label_encoder,
                            text_name, 
                            emoji_name, 
                            differing_samples):

    # Fit the embedders on the training data
    text_embedder.fit(X_train)
    emoji_embedder.fit(X_train)
    
    # Create embeddings
    X_train_text = text_embedder.transform(X_train)
    X_test_text = text_embedder.transform(X_test)
    X_train_emoji = emoji_embedder.transform(X_train)
    X_test_emoji = emoji_embedder.transform(X_test)
    
    # Create stacked input for hybrid models
    X_train_combined = np.hstack((X_train_text, X_train_emoji))
    X_test_combined = np.hstack((X_test_text, X_test_emoji))
    
    # Declare models
    model_text_only = LogisticRegression(max_iter=1000, multi_class="multinomial", solver="lbfgs")
    model_with_emoji = LogisticRegression(max_iter=1000, multi_class="multinomial", solver="lbfgs")
    
    # Fit text and hybrid models
    model_text_only.fit(X_train_text, y_train)
    model_with_emoji.fit(X_train_combined, y_train)
    
    # Make predictions for both models
    y_pred_text = model_text_only.predict(X_test_text)
    y_pred_combined = model_with_emoji.predict(X_test_combined)
    
    # Get accuracies
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
    
    # Return the difference
    return acc_combined - acc_text

def main():

    df = pd.read_csv("training.csv")
    
    X_train, X_test, y_train, y_test = train_test_split(
        df["text"], df["sentiment"], 
        test_size=0.2, 
        random_state=42
    )
    
    # Encode labels
    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train)
    y_test_encoded = label_encoder.transform(y_test)
    
    # List different embedders
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
    
    # Store results
    results = []
    differing_samples = []
    current = 0
    
    print("\nEvaluating combinations...")

    for text_name, text_embedder in text_embedders.items():
        for emoji_name, emoji_embedder in emoji_embedders.items():

            # Track progress
            current += 1
            print(f"\nProgress: {current}/25")
            print(f"Testing {text_name} + {emoji_name}")
            
            # Evaluate the pair
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

            # Save the results, print the diff
            results.append({
                'text_embedder': text_name,
                'emoji_embedder': emoji_name,
                'accuracy_difference': diff
            })
            print(f"Accuracy diff: {diff:.4f}")
    
    results_df = pd.DataFrame(results)
    
    tbl = results_df.pivot(
        index='text_embedder',
        columns='emoji_embedder',
        values='accuracy_difference'
    )
    
    # Print stats and save everything
    print(tbl)
    results_df.to_csv('embedder_evaluation_results.csv', index=False)
    differing_df = pd.DataFrame(differing_samples)
    differing_df.to_csv('differing_predictions_all.csv', index=False)
    best_result = results_df.loc[results_df['accuracy_difference'].idxmax()]
    print(f"\nBest combination:")
    print(f"Text embedder: {best_result['text_embedder']}")
    print(f"Emoji embedder: {best_result['emoji_embedder']}")
    print(f"Accuracy difference: {best_result['accuracy_difference']:.4f}")

if __name__ == "__main__":
    main() 