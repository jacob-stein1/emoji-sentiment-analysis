from abc import ABC, abstractmethod

class EmojiEmbedder(ABC):
    
    @abstractmethod
    def fit(self, texts):
        pass

    @abstractmethod
    def transform(self, texts):
        pass
