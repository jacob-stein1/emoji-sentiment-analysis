from emoji_embedder.base import EmojiEmbedder
from collections import Counter
import numpy as np
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils import extract_emojis


class OneHotEmojiEmbedder(EmojiEmbedder):
    """One-hot encoding based emoji embedder."""
    
    def __init__(self, top_n=50):
        self.top_n = top_n
        self.top_emojis = []
    
    def fit(self, texts):
        """Find top N most common emojis in the dataset."""
        emoji_counter = Counter()
        for emojis in self._extract_emojis(texts):
            emoji_counter.update(emojis)
        self.top_emojis = [emoji for emoji, _ in emoji_counter.most_common(self.top_n)]
    
    def transform(self, texts):
        """Convert texts into one-hot emoji feature vectors."""
        emoji_embedding = np.zeros((len(texts), len(self.top_emojis)))
        for i, emojis in enumerate(self._extract_emojis(texts)):
            for emoji in emojis:
                if emoji in self.top_emojis:
                    emoji_embedding[i, self.top_emojis.index(emoji)] = 1
        return emoji_embedding
