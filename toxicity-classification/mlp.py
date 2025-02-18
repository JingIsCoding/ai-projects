import torch
import torch.nn as nn
import torch.nn.functional as F


class ToxicityClassifier(nn.Module):
    def __init__(self, embedding_dim=768, hidden_dim=512, output_dim=5):
        super(ToxicityClassifier, self).__init__()
        self.embedding_dim = embedding_dim
        self.fc1 = nn.Linear(self.embedding_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)  # 5 toxicity labels
        self.dropout = nn.Dropout(0.3)

    def forward(self, embeddings, attention_mask):
        # Compute mean of embeddings only where attention mask = 1
        masked_embeddings = embeddings * attention_mask.unsqueeze(-1)
        sentence_embeddings = masked_embeddings.sum(dim=1) / attention_mask.sum(dim=1, keepdim=True)
        x = F.relu(self.fc1(sentence_embeddings))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
        return x
