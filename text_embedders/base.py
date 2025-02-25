from abc import ABC, abstractmethod

class TextEmbedder(ABC):
    
    @abstractmethod
    def fit(self, texts):
        pass

    @abstractmethod
    def transform(self, texts):
        pass