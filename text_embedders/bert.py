from transformers import BertTokenizer, BertModel
from text_embedders.base import TextEmbedder
import torch
import numpy as np

class BERTEmbedder(TextEmbedder):
    """BERT based text embedder."""
    
    def __init__(self, model_name='bert-base-uncased', max_length=128):
        self.model_name = model_name
        self.max_length = max_length
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertModel.from_pretrained(model_name)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
    
    def fit(self, texts):
        """No fitting needed as we use pretrained BERT."""
        pass
    
    def transform(self, texts):
        """Transform texts into BERT vectors."""
        texts_no_emoji = self._preprocess_texts(texts)
        vectors = []
        
        # Process in batches to avoid memory issues
        batch_size = 32
        for i in range(0, len(texts_no_emoji), batch_size):
            batch_texts = texts_no_emoji[i:i + batch_size]
            
            # Tokenize
            encoded = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors='pt'
            )
            
            # Move to device
            input_ids = encoded['input_ids'].to(self.device)
            attention_mask = encoded['attention_mask'].to(self.device)
            
            # Get BERT embeddings
            with torch.no_grad():
                outputs = self.model(input_ids, attention_mask=attention_mask)
                # Use [CLS] token embedding as sentence representation
                batch_vectors = outputs.last_hidden_state[:, 0, :].cpu().numpy()
                vectors.extend(batch_vectors)
        
        return np.array(vectors) 