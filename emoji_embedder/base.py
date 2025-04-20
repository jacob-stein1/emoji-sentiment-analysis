from abc import ABC, abstractmethod
from utils import extract_emojis

class EmojiEmbedder(ABC):
    """Base class for emoji embedders."""
    
    @abstractmethod
    def fit(self, texts):
        """Fit the embedder to the emojis in texts."""
        pass
    
    @abstractmethod
    def transform(self, texts):
        """Transform texts into emoji embeddings."""
        pass
    
    def _extract_emojis(self, texts):
        """Extract emojis from texts."""
        return [extract_emojis(text) for text in texts]
