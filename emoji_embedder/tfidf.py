from sklearn.feature_extraction.text import TfidfVectorizer
from emoji_embedder.base import EmojiEmbedder
import numpy as np

class TFIDFEmojiEmbedder(EmojiEmbedder):
    
    def __init__(self, max_features=1000):
        self.vectorizer = TfidfVectorizer(max_features=max_features)
    
    def fit(self, texts):
        seqs = self._extract_emojis(texts)
        emoji_texts = [' '.join(emojis) for emojis in seqs]
        self.vectorizer.fit(emoji_texts)
    
    def transform(self, texts):
        seqs = self._extract_emojis(texts)
        emoji_texts = [' '.join(emojis) for emojis in seqs]
        return self.vectorizer.transform(emoji_texts) 