from gensim.models import Doc2Vec
from gensim.models.doc2vec import TaggedDocument
from emoji_embedder.base import EmojiEmbedder
import numpy as np

class Doc2VecEmojiEmbedder(EmojiEmbedder):
    
    def __init__(self, vector_size=100, window=5, min_count=1, epochs=20):
        self.vector_size = vector_size
        self.window = window
        self.min_count = min_count
        self.epochs = epochs
        self.model = None
    
    def fit(self, texts):
        seqs = self._extract_emojis(texts)
        docs = [TaggedDocument(doc, [i]) for i, doc in enumerate(seqs)]        
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
        seqs = self._extract_emojis(texts)
        embeddings = []
        
        for emojis in seqs:
            embed = self.model.infer_vector(emojis)
            embeddings.append(embed)
        
        return np.array(embeddings) 