from gensim.models import Doc2Vec
from gensim.models.doc2vec import TaggedDocument
from text_embedders.base import TextEmbedder
import numpy as np

class Doc2VecTextEmbedder(TextEmbedder):
    
    def __init__(self, vector_size=100, window=5, min_count=1, epochs=20):
        self.vector_size = vector_size
        self.window = window
        self.min_count = min_count
        self.epochs = epochs
        self.model = None
    
    def fit(self, texts):
        texts_no_emoji = self._preprocess_texts(texts)
        docs = [TaggedDocument(doc.split(), [i]) for i, doc in enumerate(texts_no_emoji)]
        self.model = Doc2Vec(
            vector_size=self.vector_size,
            window=self.window,
            min_count=self.min_count,
            epochs=self.epochs)
        self.model.build_vocab(docs)
        self.model.train(
            docs, 
            total_examples=self.model.corpus_count,
            epochs=self.epochs)
    
    def transform(self, texts):
        texts_no_emoji = self._preprocess_texts(texts)
        embeddings = []
        
        for text in texts_no_emoji:
            embed = self.model.infer_vector(text.split())
            embeddings.append(embed)
        
        return np.array(embeddings) 