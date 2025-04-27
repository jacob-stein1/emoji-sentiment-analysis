from abc import ABC, abstractmethod
from utils import remove_emojis

class TextEmbedder(ABC):
    
    @abstractmethod
    def fit(self, texts):
        pass
    
    @abstractmethod
    def transform(self, texts):
        pass
    
    def _preprocess_texts(self, texts):
        return [remove_emojis(text) for text in texts]