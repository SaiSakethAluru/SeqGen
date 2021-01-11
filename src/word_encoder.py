import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import csv
# from src.encoder_transformer_block import EncoderTransformerBlock
from src.encoder_word_transformer_block import EncoderWordTransformerBlock

class WordEncoder(nn.Module):
    def __init__(
        self, 
        label_list,
        embed_size, 
        num_layers, 
        heads, 
        device, 
        forward_expansion, 
        dropout, 
        max_length,
        embed_path = '../data/glove.6B.100d.txt'
    ):
        super(WordEncoder,self).__init__()
        self.device = device
        self.labels = label_list
        ## TODO: Make this pretrained from glove
        word_embeds = pd.read_csv(filepath_or_buffer=embed_path,header=None,sep=' ',quoting=csv.QUOTE_NONE).values
        # embeds = pd.read_csv(filepath_or_buffer=embed_path,header=None,sep=' ',quoting=csv.QUOTE_NONE).values[:,1:]
        embeds = word_embeds[:,1:]
        words = word_embeds[:,:1]
        self.words = [word[0] for word in words]
        src_vocab_size,embed_size = embeds.shape
        self.embed_size = embed_size
        src_vocab_size += 2
        unknown_word = np.zeros((1, embed_size))
        pad_word = np.zeros((1,embed_size))
        # sos_word = np.zeros((1,embed_size))
        embeds = torch.from_numpy(np.concatenate([pad_word,unknown_word,embeds], axis=0).astype(np.float))

        self.word_embedding = nn.Embedding(src_vocab_size, embed_size).from_pretrained(embeds)
        self.position_embedding = nn.Embedding(max_length,embed_size)
        self.layers = nn.ModuleList(
            [
                # EncoderTransformerBlock(
                EncoderWordTransformerBlock(
                    embed_size,heads,dropout,forward_expansion,label_list    
                )
                for _ in range(num_layers)
            ]
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self,x,mask):
        N,par_len,seq_len = x.shape
        positions = torch.arange(0,seq_len).expand(N,par_len,seq_len).to(self.device)

        out = self.dropout(
            self.word_embedding(x)+self.position_embedding(positions)
        )
        label_embed = [
            self.word_embedding(torch.tensor([self.words.index(label)]).to(self.device)) for label in self.labels
        ]
        label_embed = torch.cat(label_embed,dim=0).repeat(N,1,1)
        for layer in self.layers:
            out = layer(out,out,out,label_embed,mask)
        return out
    