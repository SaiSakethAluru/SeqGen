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
        # print("decoder_block x.shape",x.shape) # N, par_len, embed_size
        # print('decoder_block value.shape',value.shape) # N, par_len, embed_size
        # print('decoder_block key.shape',key.shape) #
        # print('decoder_block encoder_word_leve_output.shape', encoder_word_level_output.shape)
        # print('decoder_block src_mask.shape',src_mask.shape)
        # print('decoder_block trg_mask.shape',trg_mask.shape)

        attention = self.attention(x, x, x, trg_mask) # N, 1, par_len, par_len
        # print('decoder_block attention.shape',attention.shape) # N, par_len, embed_size
        query = self.dropout(self.norm(attention + x))
        # print('decoder_block query.shape',query.shape)
        out = self.transformer_block(value, key, query, encoder_word_level_output, src_mask, word_level_mask)
        # print('decoder_block out.shape',out.shape) # N,par_len, embed_size
        return out