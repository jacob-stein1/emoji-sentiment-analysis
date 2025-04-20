from gensim.models import Doc2Vec
from gensim.models.doc2vec import TaggedDocument
from emoji_embedder.base import EmojiEmbedder
import numpy as np

class Doc2VecEmojiEmbedder(EmojiEmbedder):
    """Doc2Vec based emoji embedder."""
    
    def __init__(self, vector_size=100, window=5, min_count=1, epochs=20):
        self.vector_size = vector_size
        self.window = window
        self.min_count = min_count
        self.epochs = epochs
        self.model = None
    
    def fit(self, texts):
        """Fit the Doc2Vec model on emoji sequences."""
        emoji_sequences = self._extract_emojis(texts)
        # Create tagged documents for training
        tagged_docs = [TaggedDocument(doc, [i]) 
                      for i, doc in enumerate(emoji_sequences)]
        
        # Initialize and train the model
        self.model = Doc2Vec(vector_size=self.vector_size,
                           window=self.window,
                           min_count=self.min_count,
                           epochs=self.epochs)
        self.model.build_vocab(tagged_docs)
        self.model.train(tagged_docs, 
                        total_examples=self.model.corpus_count,
                        epochs=self.epochs)
    
    def transform(self, texts):
        """Transform texts into Doc2Vec emoji vectors."""
        if self.model is None:
            raise ValueError("Model not fitted. Call fit() first.")
        
        emoji_sequences = self._extract_emojis(texts)
        vectors = []
        
        for emojis in emoji_sequences:
            # Infer vector for new emoji sequence
            vector = self.model.infer_vector(emojis)
            vectors.append(vector)
        
        return np.array(vectors) 