from transformers import BertTokenizer, BertModel
from text_embedders.base import TextEmbedder
import torch
import numpy as np

class BERTEmbedder(TextEmbedder):
    
    def __init__(self, model_name='bert-base-uncased', max_length=128):
        self.model_name = model_name
        self.max_length = max_length
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertModel.from_pretrained(model_name)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
    
    def fit(self, texts):
        pass
    
    def transform(self, texts):
        texts_no_emoji = self._preprocess_texts(texts)
        embeddings = []
        
        # Process in batches to avoid memory issues
        batch_size = 32
        for i in range(0, len(texts_no_emoji), batch_size):
            batch_texts = texts_no_emoji[i:i + batch_size]
            
            encoded = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors='pt'
            )
            
            ids = encoded['input_ids'].to(self.device)
            mask = encoded['attention_mask'].to(self.device)
            
            # Get BERT embeddings
            with torch.no_grad():
                outputs = self.model(ids, attention_mask=mask)
                batch_vectors = outputs.last_hidden_state[:, 0, :].cpu().numpy()
                vectors.extend(batch_vectors)
        
        return np.array(embeddings) 