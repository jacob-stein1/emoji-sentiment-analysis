import sys
import os
import pandas as pd
import numpy as np
import random
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import warnings
warnings.filterwarnings('ignore')

# Ensure imports work correctly
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Import embedders
from text_embedders.word2vec import Word2VecTextEmbedder
from emoji_embedder.word2vec import Word2VecEmojiEmbedder
from emoji_frequency_analysis.emoji_frequencies import find_lines_with_emoji
from evaluation.neural_embedder_evaluation import TextOnlyDataset, SentimentDataset, SentimentModel

random.seed(12345)

dataset_path = "training.csv"
df = pd.read_csv(dataset_path)

def train_neural(model, train_loader, val_loader, device, use_emoji):
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(10):
        model.train()
        for batch in train_loader:
            optimizer.zero_grad()
            if use_emoji:
                text_feats, emoji_feats, labels = batch
                outputs = model(text_feats.to(device), emoji_feats.to(device))
            else:
                text_feats, labels = batch
                outputs = model(text_feats.to(device))
            loss = criterion(outputs, labels.to(device))
            loss.backward()
            optimizer.step()

    model.eval()
    preds = []
    truths = []
    with torch.no_grad():
        for batch in val_loader:
            if use_emoji:
                text_feats, emoji_feats, labels = batch
                outputs = model(text_feats.to(device), emoji_feats.to(device))
            else:
                text_feats, labels = batch
                outputs = model(text_feats.to(device))
            preds.append(outputs.argmax(dim=1).cpu())
            truths.append(labels.cpu())
    preds = torch.cat(preds)
    truths = torch.cat(truths)
    return accuracy_score(truths, preds), preds, truths

def experiment_emoji(emoji, df):
    lines = find_lines_with_emoji(emoji, df)

    if lines is None or len(lines) < 10:
        print("Not enough data for this emoji.")
        return

    random.seed(12345)
    lines_copy = lines[:]
    random.shuffle(lines_copy)
    split_index = int(len(lines_copy) * 0.8)
    train_lines = lines_copy[:split_index]
    test_lines = lines_copy[split_index:]

    X_train_texts = [t for t, _ in train_lines]
    y_train_labels = [s for _, s in train_lines]
    X_test_texts = [t for t, _ in test_lines]
    y_test_labels = [s for _, s in test_lines]

    # Encode labels
    label_encoder = LabelEncoder()
    y_train = label_encoder.fit_transform(y_train_labels)
    y_test = label_encoder.transform(y_test_labels)

    # Embedders
    text_embedder = Word2VecTextEmbedder()
    emoji_embedder = Word2VecEmojiEmbedder()
    text_embedder.fit(X_train_texts)
    emoji_embedder.fit(X_train_texts)

    X_train_text = text_embedder.transform(X_train_texts)
    X_test_text = text_embedder.transform(X_test_texts)
    X_train_emoji = emoji_embedder.transform(X_train_texts)
    X_test_emoji = emoji_embedder.transform(X_test_texts)

    if hasattr(X_train_text, 'toarray'): X_train_text = X_train_text.toarray()
    if hasattr(X_test_text, 'toarray'): X_test_text = X_test_text.toarray()
    if hasattr(X_train_emoji, 'toarray'): X_train_emoji = X_train_emoji.toarray()
    if hasattr(X_test_emoji, 'toarray'): X_test_emoji = X_test_emoji.toarray()

    # Logistic Regression models
    logreg_text = LogisticRegression(max_iter=1000)
    logreg_hybrid = LogisticRegression(max_iter=1000)

    logreg_text.fit(X_train_text, y_train)
    logreg_hybrid.fit(np.hstack((X_train_text, X_train_emoji)), y_train)

    y_pred_logreg_text = logreg_text.predict(X_test_text)
    y_pred_logreg_hybrid = logreg_hybrid.predict(np.hstack((X_test_text, X_test_emoji)))

    logreg_text_acc = accuracy_score(y_test, y_pred_logreg_text)
    logreg_hybrid_acc = accuracy_score(y_test, y_pred_logreg_hybrid)

    # Neural Net models
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_loader_hybrid = DataLoader(SentimentDataset(X_train_text, X_train_emoji, y_train), batch_size=32, shuffle=True)
    test_loader_hybrid = DataLoader(SentimentDataset(X_test_text, X_test_emoji, y_test), batch_size=32, shuffle=False)

    train_loader_text = DataLoader(TextOnlyDataset(X_train_text, y_train), batch_size=32, shuffle=True)
    test_loader_text = DataLoader(TextOnlyDataset(X_test_text, y_test), batch_size=32, shuffle=False)

    neural_text = SentimentModel(text_input_size=X_train_text.shape[1], num_classes=len(label_encoder.classes_))
    neural_hybrid = SentimentModel(text_input_size=X_train_text.shape[1], emoji_input_size=X_train_emoji.shape[1], num_classes=len(label_encoder.classes_))

    neural_text_acc, preds_text, truths = train_neural(neural_text, train_loader_text, test_loader_text, device, use_emoji=False)
    neural_hybrid_acc, preds_hybrid, _ = train_neural(neural_hybrid, train_loader_hybrid, test_loader_hybrid, device, use_emoji=True)

    # Print results
    print("\nLogistic Regression: Text-only Accuracy:", logreg_text_acc)
    print("Logistic Regression: Hybrid Accuracy:", logreg_hybrid_acc)
    print("LogReg Difference (Hybrid - Text):", logreg_hybrid_acc - logreg_text_acc)

    print("\nNeural Network: Text-only Accuracy:", neural_text_acc)
    print("Neural Network: Hybrid Accuracy:", neural_hybrid_acc)
    print("Neural Difference (Hybrid - Text):", neural_hybrid_acc - neural_text_acc)

    # Save samples where hybrid fixed text-only error (logistic regression)
    improved_samples = []
    for i in range(len(X_test_texts)):
        if y_pred_logreg_hybrid[i] == y_test[i] and y_pred_logreg_text[i] != y_test[i]:
            improved_samples.append({
                "text": X_test_texts[i],
                "true_label": label_encoder.inverse_transform([y_test[i]])[0],
                "text_only_pred": label_encoder.inverse_transform([y_pred_logreg_text[i]])[0],
                "hybrid_pred": label_encoder.inverse_transform([y_pred_logreg_hybrid[i]])[0]
            })

    improved_df = pd.DataFrame(improved_samples)
    improved_df.to_csv("improved_samples_logreg.csv", index=False)
    print(f"\nSaved {len(improved_samples)} improved samples to improved_samples_logreg.csv.")

if __name__ == "__main__":
    experiment_emoji('😭', df)
