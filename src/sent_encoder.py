import torch
import torch.nn as nn
from src.encoder_transformer_block import EncoderTransformerBlock
from src.word_encoder import WordEncoder
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
        embed_path = '../data/glove.6B.100d.txt'
    ):
        super(SentenceEncoder,self).__init__()
        self.embed_size = embed_size
        self.device = device
        self.labels = label_list
        self.word_level_encoder = WordEncoder(
            label_list, 
            embed_size, 
            num_layers, 
            heads, 
            device, 
            forward_expansion, 
            dropout, 
            max_seq_len,
            embed_path
        )
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

        self.word_embedding = nn.Embedding(src_vocab_size,embed_size).from_pretrained(embeds)   # Needed to get the label embeddings
        self.position_embedding = nn.Embedding(max_par_len,embed_size)
        self.layers = nn.ModuleList(
            [
                EncoderTransformerBlock(
                     embed_size, heads, dropout, forward_expansion, label_list
                )
                for _ in range(num_layers)
            ]
        )
        self.dropout = nn.Dropout(dropout)
    def forward(self,x,mask):
        # print("sent x.shape",x.shape)
        N,par_len,seq_len = x.shape
        # print("sent mask.shape",mask.shape)
        positions = torch.arange(0,par_len).expand(N,par_len).to(self.device)
        # Positions = N, par_len
        # print("sent positions.shape",positions.shape)
        # word_level_outputs = []
        # x = x.permute(1,0,2)
        # mask = mask.permute(1,0,2)
        # x - par_len, N, seq_len
        # for i,sent in enumerate(x):
        #     # print("sent sent.shape",sent.shape)
        #     word_level_outputs.append(
        #         self.word_level_encoder(
        #             sent.reshape(N,seq_len), mask[i].reshape(N,seq_len)
        #         )
        #     )
        word_level_outputs = self.word_level_encoder(x,mask)
        # NOTE: shape of each output tensor here should be N,seq_len,embed_size for each entry of the above list
        # NOTE: After stacking it should be N,par_len,seq_len,embed_size if everything works fine. Else adjust it.
        # word_level_outputs = torch.stack(word_level_outputs,dim=1)
        avg_word_level_outputs = torch.mean(word_level_outputs,dim=2,keepdim=True)
        #avg_word_level_outputs - N,par_len,1,embed_size
        weighted_word_level_outputs = torch.einsum('npse,npae->nps',[word_level_outputs,avg_word_level_outputs])
        #weighted_word_level_outputs - N,par_len,seq_len
        alphas = torch.softmax(weighted_word_level_outputs,dim=2)
        #alphas - N,par_len,seq_len
        combined_word_level_out = torch.einsum('npse,nps->npe',[word_level_outputs,alphas])
        #combined_word_level_out - N,par_len,embed_size

        # print("sent word_level_outputs.shape",word_level_outputs.shape)
        out = self.dropout(
            (combined_word_level_out + self.position_embedding(positions))
        )
        # print("sent out.shape",out.shape)
        # label_embed = [
        #     self.word_embedding(label) for label in self.labels
        # ]
        label_embed = [
            self.word_embedding(torch.tensor([self.words.index(label)]).to(self.device)) for label in self.labels
        ]
        # NOTE: Each entry in the above list should be 1,embed_size. If not adjust to this size
        label_embed = torch.cat(label_embed,dim=0)
        # label_embed = torch.stack(label_embed,dim=0)
        label_embed = label_embed.repeat(N,1,1)
        # print("sent label_embed.shape",label_embed.shape)
        # mask = mask.permute(1,0,2)
        # mask - N,par_len,seq_len
        mask = torch.any(mask.bool(),dim=2).int()
        # mask - N,par_len --> mask now tells only if a sentence is padded one or not.
        for layer in self.layers:
            out = layer(out,out,out,label_embed,mask)
        # print('sent out.shape',out.shape)
        # out - N,par_len,embed_size
        # word_level_output - N,par_len, seq_len, embed_size - Basically for each element in the batch, 
        # for each sentence in the abstract, we have a embed_size vector for each word
        return out, word_level_outputs


