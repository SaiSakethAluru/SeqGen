import torch 
import torch.nn as nn
from src.selfatt import SelfAttention
from src.decoder_transformer_block import DecoderTransformerBlock


class DecoderBlock(nn.Module):
    def __init__(self, embed_size, heads, forward_expansion, dropout, device):
        super(DecoderBlock, self).__init__()
        self.norm = nn.LayerNorm(embed_size)
        self.attention = SelfAttention(embed_size, heads=heads)
        self.transformer_block = DecoderTransformerBlock(
            embed_size, heads, dropout, forward_expansion
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, value, key, encoder_word_level_output, src_mask, word_level_mask, trg_mask):
        attention = self.attention(x, x, x, trg_mask) # N, 1, par_len, par_len
        query = self.dropout(self.norm(attention + x))
        out = self.transformer_block(value, key, query, encoder_word_level_output, src_mask, word_level_mask)
        return out