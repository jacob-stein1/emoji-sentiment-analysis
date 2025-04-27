from transformers import BertTokenizer, BertModel
from emoji_embedder.base import EmojiEmbedder
import torch
import numpy as np

class BERTEmojiEmbedder(EmojiEmbedder):
    
    def __init__(self, model_name='bert-base-uncased', max_length=128):
        self.model_name = model_name
        self.max_length = max_length
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertModel.from_pretrained(model_name)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
    
    def fit(self, texts):
        # Pretrained BERT
        pass
    
    def transform(self, texts):
        seqs = self._extract_emojis(texts)
        embedding = []
        
        # Batch processing
        batch_size = 32
        for i in range(0, len(seqs), batch_size):

            # Get batch
            batch_emojis = seqs[i:i + batch_size]
            batch_texts = [' '.join(emojis) for emojis in batch_emojis]
            encoded = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors='pt'
            )
            
            ids = encoded['input_ids'].to(self.device)
            mask = encoded['attention_mask'].to(self.device)
            
            with torch.no_grad():
                outputs = self.model(ids, attention_mask=mask)
                batch_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
                embedding.extend(batch_embeddings)
        
        return np.array(embedding) 