import torch
import torch.nn as nn
# from src.selfatt import SelfAttention
from src.selfatt3d import SelfAttention3D
from src.word_level_labelatt import WordLabelAttention

class LabelAttTransformerBlock(nn.Module):
    def __init__(self, embed_size, heads, dropout, forward_expansion, labels):
        super(LabelAttTransformerBlock, self).__init__()
        self.labels = labels
        label_heads = len(labels)

        # self.attention = SelfAttention(embed_size, heads)
        # self.attention = SelfAttention3D(embed_size,heads)
        # self.norm1 = nn.LayerNorm(embed_size)
        # self.label_attention = SelfAttention(embed_size, label_heads)
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
        # inputs - N,seq_len,embed_size
        # label_embed - num_labels,embed_size
        mask = mask.unsqueeze(1).unsqueeze(2)
        # print('etb x.shape', x.shape)
        # x - N,seq_len, embed_size
        # x = self.label_attention(x,label_embed,label_embed,mask)
        x = self.label_attention(label_embed, label_embed, query, mask.permute(0, 1, 3, 2, 4), att_heat_map)
        # x = self.label_attention(label_embed,x,x,mask)
        # print('etb x.shape', x.shape)
        # x - N,seq_len, embed_size
        x = self.dropout(self.label_norm(query+x))
        # print('etb x.shape', x.shape)
        # x - N,seq_len, embed_size
        forward = self.feed_forward(x)
        # print('etb forward.shape', forward.shape)
        # forward - N,seq_len,embed_size
        out = self.dropout(self.norm2(forward + x))
        # print('etb out.shape', out.shape)
        # out - N,seq_len,embed_size
        return out
