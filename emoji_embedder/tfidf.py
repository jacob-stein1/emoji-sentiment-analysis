from sklearn.feature_extraction.text import TfidfVectorizer
from emoji_embedder.base import EmojiEmbedder
import numpy as np

class TFIDFEmojiEmbedder(EmojiEmbedder):
    """TF-IDF based emoji embedder."""
    
    def __init__(self, max_features=1000):
        self.vectorizer = TfidfVectorizer(max_features=max_features)
    
    def fit(self, texts):
        """Fit the TF-IDF vectorizer on emoji sequences."""
        emoji_sequences = self._extract_emojis(texts)
        # Join emojis with spaces to create text
        emoji_texts = [' '.join(emojis) for emojis in emoji_sequences]
        self.vectorizer.fit(emoji_texts)
    
    def transform(self, texts):
        """Transform texts into TF-IDF emoji vectors."""
        emoji_sequences = self._extract_emojis(texts)
        # Join emojis with spaces to create text
        emoji_texts = [' '.join(emojis) for emojis in emoji_sequences]
        return self.vectorizer.transform(emoji_texts) 