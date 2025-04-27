from abc import ABC, abstractmethod
from utils import extract_emojis

class EmojiEmbedder(ABC):
    
    @abstractmethod
    def fit(self, texts):
        # Fit embedder to emojis in text
        pass
    
    @abstractmethod
    def transform(self, texts):
        # Transform emoji into embeddings
        pass
    
    def _extract_emojis(self, texts):
        # Extract emojis
        return [extract_emojis(text) for text in texts]
