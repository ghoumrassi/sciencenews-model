import torch
import torch.nn as nn
from transformers import AutoModel
from .utils import unfreeze_n_layers


class PretrainedTransformerClf(nn.Module):
    def __init__(
            self,
            num_classes=8,
            unfreeze=3,
            hidden_size=512,
            dropout=0.1,
            pretrained_name='distilbert-base-uncased'):
        super(PretrainedTransformerClf, self).__init__()
        # Load Pretrained transformer model
        self.transformer = AutoModel.from_pretrained(
            pretrained_name)
        # Unfreeze the last 3 layers of the DistilBERT model
        self.transformer = unfreeze_n_layers(self.transformer, unfreeze)
        # Add dense layers for classification
        self.dense1 = nn.Linear(self.transformer.config.dim, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.dense2 = nn.Linear(hidden_size, num_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        last_hidden_state = outputs.last_hidden_state
        # Pooling to get sentence representation
        sentence_representation = torch.mean(last_hidden_state, dim=1)
        x = self.dense1(sentence_representation)
        x = torch.relu(x)
        x = self.dropout(x)
        x = self.dense2(x)
        return x
