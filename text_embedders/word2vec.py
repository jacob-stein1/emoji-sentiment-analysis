from gensim.models import Word2Vec
from text_embedders.base import TextEmbedder
import numpy as np

class Word2VecTextEmbedder(TextEmbedder):
    """Word2Vec based text embedder."""
    
    def __init__(self, vector_size=100, window=5, min_count=1):
        self.vector_size = vector_size
        self.window = window
        self.min_count = min_count
        self.model = None
    
    def fit(self, texts):
        """Fit the Word2Vec model."""
        texts_no_emoji = self._preprocess_texts(texts)
        sentences = [text.split() for text in texts_no_emoji]
        self.model = Word2Vec(sentences, 
                            vector_size=self.vector_size,
                            window=self.window,
                            min_count=self.min_count)
    
    def transform(self, texts):
        """Transform texts into Word2Vec vectors."""
        if self.model is None:
            raise ValueError("Model not fitted. Call fit() first.")
        
        texts_no_emoji = self._preprocess_texts(texts)
        vectors = []
        
        for text in texts_no_emoji:
            words = text.split()
            word_vectors = []
            
            for word in words:
                if word in self.model.wv:
                    word_vectors.append(self.model.wv[word])
            
            if word_vectors:
                # Average word vectors
                vectors.append(np.mean(word_vectors, axis=0))
            else:
                # If no words found, use zero vector
                vectors.append(np.zeros(self.vector_size))
        
        return np.array(vectors) 