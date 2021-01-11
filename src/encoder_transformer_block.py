import torch
import torch.nn as nn
from src.selfatt import SelfAttention


class EncoderTransformerBlock(nn.Module):
    def __init__(self, embed_size, heads, dropout, forward_expansion, labels):
        super(EncoderTransformerBlock, self).__init__()
        self.labels = labels
        label_heads = len(labels)

        self.attention = SelfAttention(embed_size, heads)
        self.norm1 = nn.LayerNorm(embed_size)
        self.label_attention = SelfAttention(embed_size, label_heads)
        self.label_norm = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, forward_expansion * embed_size),
            nn.ReLU(),
            nn.Linear(forward_expansion * embed_size, embed_size),
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, value, key, query, label_embed, mask):
        # inputs - N,seq_len,embed_size
        # label_embed - num_labels,embed_size
        # print('etb value.shape', value.shape)
        # print('etb key.shape', key.shape)
        # print('etb query.shape', query.shape)
        # print('etb label_embed.shape', label_embed.shape)
        mask = mask.unsqueeze(1).unsqueeze(2)
        # print('etb mask.shape', mask.shape)
        attention = self.attention(value, key, query, mask)
        # print('etb attention.shape', attention.shape)
        # attention - N,seq_len, embed_size
        # Add skip connection, run through normalization and finally dropout
        # x - N,seq_len, embed_size
        x = self.dropout(self.norm1(attention + query))
        # print('etb x.shape', x.shape)
        # x - N,seq_len, embed_size
        # x = self.label_attention(x,label_embed,label_embed,mask)
        x = self.label_attention(label_embed, label_embed, x, mask.permute(0, 1, 3, 2))
        # x = self.label_attention(label_embed,x,x,mask)
        # print('etb x.shape', x.shape)
        # x - N,seq_len, embed_size
        x = self.dropout(self.label_norm(x))
        # print('etb x.shape', x.shape)
        # x - N,seq_len, embed_size
        forward = self.feed_forward(x)
        # print('etb forward.shape', forward.shape)
        # forward - N,seq_len,embed_size
        out = self.dropout(self.norm2(forward + x))
        # print('etb out.shape', out.shape)
        # out - N,seq_len,embed_size
        return out
