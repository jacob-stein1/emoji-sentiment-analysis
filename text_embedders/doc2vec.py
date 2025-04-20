from gensim.models import Doc2Vec
from gensim.models.doc2vec import TaggedDocument
from text_embedders.base import TextEmbedder
import numpy as np

class Doc2VecTextEmbedder(TextEmbedder):
    """Doc2Vec based text embedder."""
    
    def __init__(self, vector_size=100, window=5, min_count=1, epochs=20):
        self.vector_size = vector_size
        self.window = window
        self.min_count = min_count
        self.epochs = epochs
        self.model = None
    
    def fit(self, texts):
        """Fit the Doc2Vec model."""
        texts_no_emoji = self._preprocess_texts(texts)
        # Create tagged documents for training
        tagged_docs = [TaggedDocument(doc.split(), [i]) 
                      for i, doc in enumerate(texts_no_emoji)]
        
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
        """Transform texts into Doc2Vec vectors."""
        if self.model is None:
            raise ValueError("Model not fitted. Call fit() first.")
        
        texts_no_emoji = self._preprocess_texts(texts)
        vectors = []
        
        for text in texts_no_emoji:
            # Infer vector for new document
            vector = self.model.infer_vector(text.split())
            vectors.append(vector)
        
        return np.array(vectors) 