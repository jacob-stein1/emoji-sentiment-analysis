from gensim.models import Word2Vec
from emoji_embedder.base import EmojiEmbedder
import numpy as np

class Word2VecEmojiEmbedder(EmojiEmbedder):
    
    def __init__(self, vector_size=100, window=5, min_count=1):
        self.vector_size = vector_size
        self.window = window
        self.min_count = min_count
        self.model = None
    
    def fit(self, texts):
        emoji_sequences = self._extract_emojis(texts)
        self.model = Word2Vec(
            emoji_sequences,
            vector_size=self.vector_size,
            window=self.window,
            min_count=self.min_count)
    
    def transform(self, texts):
        
        seqs = self._extract_emojis(texts)
        embeddings = []
        
        for emojis in seqs:
            emoji_vectors = []
            for emoji in emojis:
                if emoji in self.model.wv:
                    emoji_vectors.append(self.model.wv[emoji])
            
            if emoji_vectors:
                embeddings.append(np.mean(emoji_vectors, axis=0))
            else:
                embeddings.append(np.zeros(self.vector_size))
        
        return np.array(embeddings) 