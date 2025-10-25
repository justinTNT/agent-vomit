
import torch
import torch.nn as nn

class SimpleSeq2Seq(nn.Module):
    def __init__(self, vocab_size, d_model, n_heads, n_layers):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model, n_heads),
            n_layers
        )
        self.decoder = nn.Linear(d_model, vocab_size)
        
    def forward(self, x):
        x = self.embedding(x)
        x = self.encoder(x)
        return self.decoder(x.mean(dim=1))
