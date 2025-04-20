from gensim.models import Word2Vec
from emoji_embedder.base import EmojiEmbedder
import numpy as np

class SkipgramEmojiEmbedder(EmojiEmbedder):
    """Skip-gram based emoji embedder."""
    
    def __init__(self, vector_size=100, window=5, min_count=1, epochs=20):
        self.vector_size = vector_size
        self.window = window
        self.min_count = min_count
        self.epochs = epochs
        self.model = None
    
    def fit(self, texts):
        """Fit the Skip-gram model on emojis."""
        emoji_sequences = self._extract_emojis(texts)
        self.model = Word2Vec(emoji_sequences,
                            vector_size=self.vector_size,
                            window=self.window,
                            min_count=self.min_count,
                            epochs=self.epochs,
                            sg=1)  # Use Skip-gram architecture
    
    def transform(self, texts):
        """Transform texts into Skip-gram emoji vectors."""
        if self.model is None:
            raise ValueError("Model not fitted. Call fit() first.")
        
        emoji_sequences = self._extract_emojis(texts)
        vectors = []
        
        for emojis in emoji_sequences:
            emoji_vectors = []
            for emoji in emojis:
                if emoji in self.model.wv:
                    emoji_vectors.append(self.model.wv[emoji])
            
            if emoji_vectors:
                # Average emoji vectors
                vectors.append(np.mean(emoji_vectors, axis=0))
            else:
                # If no emojis found, use zero vector
                vectors.append(np.zeros(self.vector_size))
        
        return np.array(vectors) 