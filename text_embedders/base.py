from abc import ABC, abstractmethod
from utils import remove_emojis

class TextEmbedder(ABC):
    """Base class for text embedders."""
    
    @abstractmethod
    def fit(self, texts):
        """Fit the embedder to the texts."""
        pass
    
    @abstractmethod
    def transform(self, texts):
        """Transform texts into embeddings."""
        pass
    
    def _preprocess_texts(self, texts):
        """Remove emojis from texts."""
        return [remove_emojis(text) for text in texts]