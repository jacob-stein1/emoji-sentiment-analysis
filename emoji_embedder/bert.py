from transformers import BertTokenizer, BertModel
from emoji_embedder.base import EmojiEmbedder
import torch
import numpy as np

class BERTEmojiEmbedder(EmojiEmbedder):
    """BERT based emoji embedder."""
    
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
        """Transform texts into BERT emoji vectors."""
        emoji_sequences = self._extract_emojis(texts)
        vectors = []
        
        # Process in batches to avoid memory issues
        batch_size = 32
        for i in range(0, len(emoji_sequences), batch_size):
            batch_emojis = emoji_sequences[i:i + batch_size]
            # Join emojis with spaces to create text
            batch_texts = [' '.join(emojis) for emojis in batch_emojis]
            
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
                # Use [CLS] token embedding as emoji sequence representation
                batch_vectors = outputs.last_hidden_state[:, 0, :].cpu().numpy()
                vectors.extend(batch_vectors)
        
        return np.array(vectors) 