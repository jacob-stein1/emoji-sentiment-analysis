import torch
import torch.nn as nn

class SentimentModel(nn.Module):
    def __init__(self, text_input_size, emoji_input_size=None, num_classes=3):
        super(SentimentModel, self).__init__()
        
        self.text_cnn = nn.Sequential(
            nn.Conv1d(1, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Flatten()
        )
        
        cnn_output_size = 256 * (text_input_size // 4)
        
        self.use_emoji = emoji_input_size is not None
        if self.use_emoji:
            self.emoji_lstm = nn.LSTM(
                input_size=emoji_input_size,
                hidden_size=256,
                num_layers=2,
                batch_first=True,
                bidirectional=True
            )
            lstm_output_size = 512
            classifier_input_size = cnn_output_size + lstm_output_size
        else:
            classifier_input_size = cnn_output_size
        
        self.classifier = nn.Sequential(
            nn.Linear(classifier_input_size, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, text_features, emoji_features=None):
        text_features = text_features.unsqueeze(1)
        text_features = self.text_cnn(text_features)
        
        if self.use_emoji and emoji_features is not None:
            if len(emoji_features.shape) == 2:
                emoji_features = emoji_features.unsqueeze(1)
            lstm_out, _ = self.emoji_lstm(emoji_features)
            emoji_features = lstm_out[:, -1, :]
            combined = torch.cat([text_features, emoji_features], dim=1)
        else:
            combined = text_features
        
        output = self.classifier(combined)
        return output 