import sys
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder

# Ensure imports work correctly by adding parent directories to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Import embedders and utilities
from text_embedders.tfidf import TFIDFTextEmbedder
from emoji_embedder.onehot import OneHotEmojiEmbedder
from utils import analyze_pmf_difference

# Load dataset
dataset_path = "training.csv"
df = pd.read_csv(dataset_path)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(df["text"], df["sentiment"], test_size=0.2, random_state=42)

# Initialize embedders
text_embedder = TFIDFTextEmbedder()
emoji_embedder = OneHotEmojiEmbedder()

# Train embedders
text_embedder.fit(X_train)
emoji_embedder.fit(X_train)

# Transform text and emoji embeddings
X_train_text = text_embedder.transform(X_train).toarray()
X_test_text = text_embedder.transform(X_test).toarray()
X_train_emoji = emoji_embedder.transform(X_train)
X_test_emoji = emoji_embedder.transform(X_test)

# Combine features
X_train_combined = np.hstack((X_train_text, X_train_emoji))
X_test_combined = np.hstack((X_test_text, X_test_emoji))

# Encode labels
label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)
y_test_encoded = label_encoder.transform(y_test)

# Train models
model_with_emoji = LogisticRegression(max_iter=1000, multi_class="multinomial", solver="lbfgs")
model_with_emoji.fit(X_train_combined, y_train_encoded)

model_without_emoji = LogisticRegression(max_iter=1000, multi_class="multinomial", solver="lbfgs")
model_without_emoji.fit(X_train_text, y_train_encoded)

# Function to compare PMFs
def compare_pmfs(sample_text):
    """Compares sentiment PMFs for text with and without emojis."""
    
    sample_text_tfidf = text_embedder.transform([sample_text]).toarray()
    sample_emoji_embedding = emoji_embedder.transform([sample_text])

    sample_with_emoji = np.hstack((sample_text_tfidf, sample_emoji_embedding))

    pmf_with_emoji = model_with_emoji.predict_proba(sample_with_emoji)[0]
    pmf_without_emoji = model_without_emoji.predict_proba(sample_text_tfidf)[0]

    pmf_df = pd.DataFrame({
        "with_emoji": pmf_with_emoji,
        "without_emoji": pmf_without_emoji
    }, index=label_encoder.classes_)

    return pmf_df

# Test on a sample text
sample_text = "I love this! 😊"
pmf_df = compare_pmfs(sample_text)

# Analyze and plot results
analyze_pmf_difference(pmf_df)
