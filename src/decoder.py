import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import csv
from src.decoder_block import DecoderBlock


class Decoder(nn.Module):
    def __init__(
            self,
            label_list,
            embed_size,
            num_layers,
            heads,
            forward_expansion,
            dropout,
            device,
            max_par_len,
            max_seq_len,
            embed_path = '../data/glove.6B.100d.txt'
    ):
        super(Decoder, self).__init__()
        self.device = device
        word_embeds = pd.read_csv(filepath_or_buffer=embed_path,header=None,sep=' ',quoting=csv.QUOTE_NONE).values
        words = word_embeds[:,:1]
        words = [word[0] for word in words]
        embeds = word_embeds[:,1:]
        label_indices = [words.index(label) for label in label_list]
        embeds = embeds[label_indices,:] # take only embeddings of label words
        trg_vocab_size,embed_size = embeds.shape
        self.embed_size = embed_size
        trg_vocab_size += 2     # Pad and <sos> extra
        pad_word = np.zeros((1,embed_size))
        sos_word = np.zeros((1,embed_size))
        embeds = torch.from_numpy(np.concatenate([pad_word,sos_word,embeds], axis=0).astype(np.double))
        self.word_embedding = nn.Embedding(trg_vocab_size, embed_size).from_pretrained(embeds)
        self.position_embedding = nn.Embedding(max_par_len, embed_size)

        self.layers = nn.ModuleList(
            [
                DecoderBlock(embed_size, heads, forward_expansion, dropout, device)
                for _ in range(num_layers)
            ]
        )

        self.fc_out = nn.Linear(embed_size, len(label_list)+2)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, enc_out, enc_word_out, src_mask, word_level_mask, trg_mask):
        N, par_len = x.shape
        positions = torch.arange(0, par_len).expand(N, par_len).to(self.device)
        x = self.dropout((self.word_embedding(x) + self.position_embedding(positions)))
        for layer in self.layers:
            x = layer(x, enc_out, enc_out, enc_word_out, src_mask, word_level_mask, trg_mask)

        out = self.fc_out(x)
        return out
