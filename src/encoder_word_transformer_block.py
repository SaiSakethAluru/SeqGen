import torch
import torch.nn as nn
# from src.selfatt import SelfAttention
from src.selfatt3d import SelfAttention3D
from src.word_level_labelatt import WordLabelAttention

class EncoderWordTransformerBlock(nn.Module):
    def __init__(self, embed_size, heads, dropout, forward_expansion, labels):
        super(EncoderWordTransformerBlock, self).__init__()
        self.labels = labels
        label_heads = len(labels)

        self.attention = SelfAttention3D(embed_size,heads)
        self.norm1 = nn.LayerNorm(embed_size)
        self.label_attention = WordLabelAttention(embed_size,label_heads)
        self.label_norm = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, forward_expansion * embed_size),
            nn.ReLU(),
            nn.Linear(forward_expansion * embed_size, embed_size),
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, value, key, query, label_embed, mask):
        mask = mask.unsqueeze(1).unsqueeze(2)
        attention = self.attention(value, key, query, mask)
        # Add skip connection, run through normalization and finally dropout
        x = self.dropout(self.norm1(attention + query))
        x = self.label_attention(label_embed, label_embed, x, mask.permute(0, 1, 3, 2, 4))
        x = self.dropout(self.label_norm(x))
        forward = self.feed_forward(x)
        out = self.dropout(self.norm2(forward + x))
        return out
