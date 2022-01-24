import torch
import torch.nn as nn
from src.encoder_transformer_block import EncoderTransformerBlock
# from src.word_encoder import WordEncoder
from src.label_att_transformer_block import LabelAttTransformerBlock
from transformers import AutoModel
import pandas as pd
import numpy as np
import csv
class SentenceEncoder(nn.Module):
    def __init__(
        self,
        label_list,
        embed_size, 
        num_layers, 
        heads, 
        device, 
        forward_expansion, 
        dropout, 
        max_par_len,
        max_seq_len,
        bert_model="allenai/scibert_scivocab_uncased"
    ):
        super(SentenceEncoder,self).__init__()
        self.embed_size = embed_size
        self.device = device
        self.labels = label_list
        self.word_level_encoder = AutoModel.from_pretrained(bert_model)

        for p in self.word_level_encoder.parameters():
            p.requires_grad = False

        self.reshape_fc1 = nn.Linear(768,3*embed_size)
        self.reshape_fc_pool1 = nn.Linear(768,3*embed_size)

        self.reshape_fc2 = nn.Linear(3*embed_size, embed_size)
        self.reshape_fc_pool2 = nn.Linear(3*embed_size, embed_size)

        self.word_embedding = nn.Embedding(len(label_list),embed_size)

        self.position_embedding = nn.Embedding(max_par_len,embed_size)
        self.word_label_layers = nn.ModuleList(
            [
                LabelAttTransformerBlock(
                    embed_size, heads, dropout, forward_expansion, label_list
                )
                for _ in range(num_layers)
            ]
        )
        self.layers = nn.ModuleList(
            [
                EncoderTransformerBlock(
                     embed_size, heads, dropout, forward_expansion, label_list
                )
                for _ in range(num_layers)
            ]
        )
        self.dropout = nn.Dropout(dropout)
    def forward(self,x,mask, att_heat_map=False):
        N,par_len,seq_len = x.shape
        positions = torch.arange(0,par_len).expand(N,par_len).to(self.device)  
        
        word_level_bert_output = self.word_level_encoder(
            input_ids = x.reshape(N*par_len,seq_len),
            attention_mask = mask.reshape(N*par_len,seq_len)
        )

        word_level_outputs = word_level_bert_output.last_hidden_state
        pooled_output = word_level_bert_output.pooler_output
        word_level_outputs = word_level_outputs.reshape(N,par_len,word_level_outputs.shape[1],word_level_outputs.shape[2])
        pooled_output = pooled_output.reshape(N,par_len,-1)


        word_level_outputs = self.reshape_fc1(word_level_outputs)    
        pooled_output = self.reshape_fc_pool1(pooled_output)        

        word_level_outputs = self.reshape_fc2(word_level_outputs)   
        pooled_output = self.reshape_fc_pool2(pooled_output)        

        label_embed = [
            self.word_embedding(torch.Tensor([self.labels.index(label)]).to(self.device).long()) for label in self.labels
        ]
        label_embed = torch.cat(label_embed,dim=0)
        label_embed = label_embed.repeat(N,1,1)
        for layer in self.word_label_layers:
            word_level_outputs = layer(word_level_outputs, label_embed,mask, att_heat_map)
        
        out = word_level_outputs[:,:,0,:]

        out = self.dropout(
            (out + self.position_embedding(positions))
        )

        mask = torch.any(mask.bool(),dim=2).int()

        for layer in self.layers:
            out = layer(out,out,out,label_embed,mask, att_heat_map)
        
        return out


