import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from text_embedders.base import TextEmbedder

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils import remove_emojis

class TFIDFTextEmbedder(TextEmbedder):
    """TF-IDF based text embedder."""
    
    def __init__(self, max_features=1000):
        self.vectorizer = TfidfVectorizer(max_features=max_features)
    
    def fit(self, texts):
        """Fit the TF-IDF vectorizer."""
        texts_no_emoji = self._preprocess_texts(texts)
        self.vectorizer.fit(texts_no_emoji)
    
    def transform(self, texts):
        """Transform texts into TF-IDF vectors."""
        texts_no_emoji = self._preprocess_texts(texts)
        return self.vectorizer.transform(texts_no_emoji)
