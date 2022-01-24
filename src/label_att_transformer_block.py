import torch
import torch.nn as nn
from src.selfatt3d import SelfAttention3D
from src.word_level_labelatt import WordLabelAttention

class LabelAttTransformerBlock(nn.Module):
    def __init__(self, embed_size, heads, dropout, forward_expansion, labels):
        super(LabelAttTransformerBlock, self).__init__()
        self.labels = labels
        label_heads = len(labels)

        self.label_attention = WordLabelAttention(embed_size,label_heads)
        self.label_norm = nn.LayerNorm(embed_size)
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, forward_expansion * embed_size),
            nn.ReLU(),
            nn.Linear(forward_expansion * embed_size, embed_size),
        )
        self.norm2 = nn.LayerNorm(embed_size)

        self.dropout = nn.Dropout(dropout)

    def forward(self, query, label_embed, mask, att_heat_map=False):
        mask = mask.unsqueeze(1).unsqueeze(2)
        x = self.label_attention(label_embed, label_embed, query, mask.permute(0, 1, 3, 2, 4), att_heat_map)
        x = self.dropout(self.label_norm(query+x))
        forward = self.feed_forward(x)
        out = self.dropout(self.norm2(forward + x))
        return out
