import torch
import torch.nn as nn
from src.selfatt import SelfAttention
from src.decoder_word_crossatt import CrossAttention


class DecoderTransformerBlock(nn.Module):
    def __init__(self, embed_size, heads, dropout, forward_expansion):
        super(DecoderTransformerBlock, self).__init__()
        self.attention = SelfAttention(embed_size, heads)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)
        self.norm3 = nn.LayerNorm(embed_size)
        self.norm4 = nn.LayerNorm(embed_size)
        self.enc_sent_attention = CrossAttention(embed_size, heads)
        self.feed_forward1 = nn.Sequential(
            nn.Linear(embed_size, forward_expansion * embed_size),
            nn.ReLU(),
            nn.Linear(forward_expansion * embed_size, embed_size),
        )
        self.feed_forward2 = nn.Sequential(
            nn.Linear(embed_size, forward_expansion * embed_size),
            nn.ReLU(),
            nn.Linear(forward_expansion * embed_size, embed_size),
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, value, key, query, encoder_word_level_output, mask, word_level_mask):
        # encoder_word_level should be N,embed_size
        par_mask = torch.any(mask.bool(),dim=2).int().unsqueeze(1).unsqueeze(2)  # N, 1,1,par_len
        attention = self.attention(value, key, query, par_mask)  
        # Add skip connection, run through normalization and finally dropout
        x = self.dropout(self.norm1(attention + query))

        forward = self.feed_forward1(x)
        x = self.dropout(self.norm2(forward+x))
        
        # include word level outputs of encoder here
        encoder_word_attentions = self.enc_sent_attention(encoder_word_level_output, encoder_word_level_output, x, word_level_mask) 
        x = self.dropout(self.norm3(x + encoder_word_attentions))

        forward = self.feed_forward2(x)
        out = self.dropout(self.norm4(forward + x))
        return out
