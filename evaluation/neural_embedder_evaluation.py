import sys
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
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

class SentimentDataset(Dataset):
    def __init__(self, text_features, emoji_features, labels):
        self.text_features = torch.FloatTensor(text_features)
        self.emoji_features = torch.FloatTensor(emoji_features)
        self.labels = torch.LongTensor(labels)
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return self.text_features[idx], self.emoji_features[idx], self.labels[idx]

class SentimentModel(nn.Module):
    def __init__(self, text_input_size, emoji_input_size=None, num_classes=3):
        super(SentimentModel, self).__init__()
        
        self.text_cnn = nn.Sequential(
            nn.Conv1d(1, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Flatten()
        )
        
        cnn_output_size = 256 * (text_input_size // 4)
        
        self.use_emoji = emoji_input_size is not None
        if self.use_emoji:
            self.emoji_lstm = nn.LSTM(
                input_size=emoji_input_size,
                hidden_size=256,
                num_layers=2,
                batch_first=True,
                bidirectional=True
            )
            lstm_output_size = 512
            classifier_input_size = cnn_output_size + lstm_output_size
        else:
            classifier_input_size = cnn_output_size
        
        self.classifier = nn.Sequential(
            nn.Linear(classifier_input_size, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, text_features, emoji_features=None):
        text_features = text_features.unsqueeze(1)
        text_features = self.text_cnn(text_features)
        
        if self.use_emoji and emoji_features is not None:
            if len(emoji_features.shape) == 2:
                emoji_features = emoji_features.unsqueeze(1)
            lstm_out, _ = self.emoji_lstm(emoji_features)
            emoji_features = lstm_out[:, -1, :]
            combined = torch.cat([text_features, emoji_features], dim=1)
        else:
            combined = text_features
        
        output = self.classifier(combined)
        return output

class TextOnlyDataset(Dataset):
    def __init__(self, text_features, labels):
        self.text_features = torch.FloatTensor(text_features)
        self.labels = torch.LongTensor(labels)
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return self.text_features[idx], self.labels[idx]

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, device):
    best_val_acc = 0.0
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for text_features, emoji_features, labels in train_loader:
            text_features = text_features.to(device)
            emoji_features = emoji_features.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(text_features, emoji_features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
        
        train_acc = 100 * train_correct / train_total
        
        model.eval()
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for text_features, emoji_features, labels in val_loader:
                text_features = text_features.to(device)
                emoji_features = emoji_features.to(device)
                labels = labels.to(device)
                
                outputs = model(text_features, emoji_features)
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        val_acc = 100 * val_correct / val_total
        
        print(f'Epoch [{epoch+1}/{num_epochs}], '
              f'Train Loss: {train_loss/len(train_loader):.4f}, '
              f'Train Acc: {train_acc:.2f}%, '
              f'Val Acc: {val_acc:.2f}%')
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
    
    return best_val_acc

def train_text_only_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, device):
    best_val_acc = 0.0
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for text_features, labels in train_loader:
            text_features = text_features.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(text_features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
        
        train_acc = 100 * train_correct / train_total
        
        model.eval()
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for text_features, labels in val_loader:
                text_features = text_features.to(device)
                labels = labels.to(device)
                
                outputs = model(text_features)
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        val_acc = 100 * val_correct / val_total
        
        print(f'Epoch [{epoch+1}/{num_epochs}], '
              f'Train Loss: {train_loss/len(train_loader):.4f}, '
              f'Train Acc: {train_acc:.2f}%, '
              f'Val Acc: {val_acc:.2f}%')
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
    
    return best_val_acc

def evaluate_embedder_pair(text_embedder, emoji_embedder, X_train, X_test, y_train, y_test, label_encoder, text_name, emoji_name, differing_samples):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    text_embedder.fit(X_train)
    emoji_embedder.fit(X_train)
    
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
    
    if len(X_train_text.shape) == 1:
        X_train_text = X_train_text.reshape(-1, 1)
        X_test_text = X_test_text.reshape(-1, 1)
    if len(X_train_emoji.shape) == 1:
        X_train_emoji = X_train_emoji.reshape(-1, 1)
        X_test_emoji = X_test_emoji.reshape(-1, 1)
    
    train_dataset = SentimentDataset(X_train_text, X_train_emoji, y_train)
    test_dataset = SentimentDataset(X_test_text, X_test_emoji, y_test)
    train_dataset_text = TextOnlyDataset(X_train_text, y_train)
    test_dataset_text = TextOnlyDataset(X_test_text, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    train_loader_text = DataLoader(train_dataset_text, batch_size=32, shuffle=True)
    test_loader_text = DataLoader(test_dataset_text, batch_size=32, shuffle=False)
    
    hybrid_model = SentimentModel(text_input_size=X_train_text.shape[1], emoji_input_size=X_train_emoji.shape[1], num_classes=len(label_encoder.classes_)).to(device)
    text_only_model = SentimentModel(text_input_size=X_train_text.shape[1], num_classes=len(label_encoder.classes_)).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer_hybrid = optim.Adam(hybrid_model.parameters(), lr=0.001)
    optimizer_text = optim.Adam(text_only_model.parameters(), lr=0.001)
    
    print("\nTraining hybrid model (text + emoji)...")
    hybrid_acc = train_model(hybrid_model, train_loader, test_loader, criterion, optimizer_hybrid, 10, device)
    
    print("\nTraining text-only model...")
    text_only_acc = train_text_only_model(text_only_model, train_loader_text, test_loader_text, criterion, optimizer_text, 10, device)
    
    # Evaluate and collect misclassifications
    hybrid_model.eval()
    text_only_model.eval()
    X_test_text_tensor = torch.FloatTensor(X_test_text).to(device)
    X_test_emoji_tensor = torch.FloatTensor(X_test_emoji).to(device)
    
    with torch.no_grad():
        hybrid_outputs = hybrid_model(X_test_text_tensor, X_test_emoji_tensor)
        text_only_outputs = text_only_model(X_test_text_tensor)
    
    hybrid_preds = hybrid_outputs.argmax(dim=1).cpu().numpy()
    text_only_preds = text_only_outputs.argmax(dim=1).cpu().numpy()
    
    for i in range(len(X_test)):
        if hybrid_preds[i] != text_only_preds[i]:
            differing_samples.append({
                'text_embedder': text_name,
                'emoji_embedder': emoji_name,
                'text': X_test.iloc[i],
                'text_only_prediction': label_encoder.inverse_transform([text_only_preds[i]])[0],
                'hybrid_prediction': label_encoder.inverse_transform([hybrid_preds[i]])[0],
                'true_class': label_encoder.inverse_transform([y_test[i]])[0]
            })
    
    return hybrid_acc, text_only_acc

def main():
    print("Loading dataset...")
    df = pd.read_csv("training.csv")
    
    X_train, X_test, y_train, y_test = train_test_split(df["text"], df["sentiment"], test_size=0.2, random_state=42)
    
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
            
            try:
                hybrid_acc, text_only_acc = evaluate_embedder_pair(
                    text_embedder, 
                    emoji_embedder,
                    X_train, X_test,
                    y_train_encoded, y_test_encoded,
                    label_encoder,
                    text_name,
                    emoji_name,
                    differing_samples
                )
                accuracy_diff = hybrid_acc - text_only_acc
                results.append({
                    'text_embedder': text_name,
                    'emoji_embedder': emoji_name,
                    'hybrid_accuracy': hybrid_acc,
                    'text_only_accuracy': text_only_acc,
                    'accuracy_difference': accuracy_diff
                })
                print(f"Hybrid Accuracy: {hybrid_acc:.2f}%")
                print(f"Text-only Accuracy: {text_only_acc:.2f}%")
                print(f"Accuracy Difference: {accuracy_diff:.2f}%")
            except Exception as e:
                print(f"Error with {text_name} + {emoji_name}: {str(e)}")
    
    results_df = pd.DataFrame(results)
    results_df.to_csv('neural_embedder_evaluation_results.csv', index=False)
    
    if differing_samples:
        differing_df = pd.DataFrame(differing_samples)
        differing_df.to_csv('differing_predictions_neural.csv', index=False)
        print(f"Saved {len(differing_samples)} differing predictions to 'differing_predictions_neural.csv'")

if __name__ == "__main__":
    main()
