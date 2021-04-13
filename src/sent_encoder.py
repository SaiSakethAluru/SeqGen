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
        batch_size,
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
        self.batch_size = batch_size
        print(label_list)
        # self.word_level_encoder = WordEncoder(
        #     label_list, 
        #     embed_size, 
        #     num_layers, 
        #     heads, 
        #     device, 
        #     forward_expansion, 
        #     dropout, 
        #     max_seq_len,
        #     embed_path
        # )
        self.word_level_encoder = AutoModel.from_pretrained(bert_model)

        for p in self.word_level_encoder.parameters():
            p.requires_grad = False

        ## Should we keep a fc layer to reshape the output layers?
        
        self.reshape_fc1 = nn.Linear(768,3*embed_size)
        self.reshape_fc_pool1 = nn.Linear(768,3*embed_size)

        self.reshape_fc2 = nn.Linear(3*embed_size, embed_size)
        self.reshape_fc_pool2 = nn.Linear(3*embed_size, embed_size)

        # ## TODO: Make this pretrained from glove
        # word_embeds = pd.read_csv(filepath_or_buffer=embed_path,header=None,sep=' ',quoting=csv.QUOTE_NONE).values
        # # embeds = pd.read_csv(filepath_or_buffer=embed_path,header=None,sep=' ',quoting=csv.QUOTE_NONE).values[:,1:]
        # embeds = word_embeds[:,1:]
        # words = word_embeds[:,:1]
        # self.words = [word[0] for word in words]
        # src_vocab_size,embed_size = embeds.shape
        # self.embed_size = embed_size
        # src_vocab_size += 2
        # unknown_word = np.zeros((1, embed_size))
        # pad_word = np.zeros((1,embed_size))
        # # sos_word = np.zeros((1,embed_size))
        # embeds = torch.from_numpy(np.concatenate([pad_word,unknown_word,embeds], axis=0).astype(np.float))

        # self.word_embedding = nn.Embedding(src_vocab_size,embed_size).from_pretrained(embeds)   # Needed to get the label embeddings
        
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
        ## Sentence level transformer layers
        # self.layers = nn.ModuleList(
        #     [
        #         EncoderTransformerBlock(
        #              embed_size, heads, dropout, forward_expansion, label_list
        #         )
        #         for _ in range(num_layers)
        #     ]
        # )
        ## Use for baseline sentence level LSTM
        self.sent_level_lstm = nn.LSTM(embed_size,embed_size, num_layers=num_layers, batch_first=True,dropout=dropout,bidirectional=True)
        self.sent_level_lstm_cell = torch.randn(num_layers*2, self.batch_size, embed_size).to(device)
        self.sent_level_lstm_hidden = torch.randn(num_layers*2, self.batch_size, embed_size).to(device)
        self.dropout = nn.Dropout(dropout)
    def forward(self,x,mask, att_heat_map=False):
        N,par_len,seq_len = x.shape
        positions = torch.arange(0,par_len).expand(N,par_len).to(self.device)   #position - N,par_len
        
        # should pass batch_size x seq_len vectors to transformers model. Hence reshape the x into 2D
        word_level_bert_output = self.word_level_encoder(
            input_ids = x.reshape(N*par_len,seq_len),
            attention_mask = mask.reshape(N*par_len,seq_len)
        )   # Output shape = (last layer outputs - N*par_len,seq_len, hidden_size; pooled_output - N*par_len,hidden_size)

        word_level_outputs = word_level_bert_output.last_hidden_state
        pooled_output = word_level_bert_output.pooler_output
        # print('type(word_level_outputs)',type(word_level_outputs))
        # print('word_level_outputs',word_level_outputs)
        # print('type(pooled_output)',type(pooled_output))
        # print('pooled_output',pooled_output)
        # assert False
        word_level_outputs = word_level_outputs.reshape(N,par_len,word_level_outputs.shape[1],word_level_outputs.shape[2])
        pooled_output = pooled_output.reshape(N,par_len,-1)


        word_level_outputs = self.reshape_fc1(word_level_outputs)    # word_level_outputs -> N,par_len,seq_len,3*embed_size
        pooled_output = self.reshape_fc_pool1(pooled_output)         # pooled_output -> N,par_len,3*embed_size

        word_level_outputs = self.reshape_fc2(word_level_outputs)    # word_level_outputs -> N,par_len,seq_len,embed_size
        pooled_output = self.reshape_fc_pool2(pooled_output)        # pooled_output -> N,par_len,embed_size
        # return pooled_output

        # # print("sent word_level_outputs.shape",word_level_outputs.shape)
        # # print("sent out.shape",out.shape)
        # # label_embed = [
        # #     self.word_embedding(label) for label in self.labels
        # # ]


        label_embed = [
            self.word_embedding(torch.Tensor([self.labels.index(label)]).to(self.device).long()) for label in self.labels
        ]
        # # NOTE: Each entry in the above list should be 1,embed_size. If not adjust to this size
        label_embed = torch.cat(label_embed,dim=0)
        # # label_embed = torch.stack(label_embed,dim=0)
        label_embed = label_embed.repeat(N,1,1)
        # # print("sent label_embed.shape",label_embed.shape)
        # # mask = mask.permute(1,0,2)
        # # mask - N,par_len,seq_len
        ## Uncomment for full model. Commented out for ablation.
        for layer in self.word_label_layers:
            word_level_outputs = layer(word_level_outputs, label_embed,mask, att_heat_map)
        
        out = word_level_outputs[:,:,0,:]
        
        ## For ablation study
        # return out

        ## Baseline -> LSTM in sentence level instead of Transformer
        lstm_out,(lstm_last_hidden, lstm_last_cell) = self.sent_level_lstm(out,(self.sent_level_lstm_hidden,self.sent_level_lstm_cell))

        return lstm_last_hidden
        ## Sentence level transformer

        # CODE: sent level encoder begin
        # out = self.dropout(
        #     (out + self.position_embedding(positions))
        # )
        # mask = torch.any(mask.bool(),dim=2).int()
        # CODE: sent level encoder end
        # # mask - N,par_len --> mask now tells only if a sentence is padded one or not.

        # label_embed = None
        # CODE: Sent level encoder begin
        # for layer in self.layers:
            # out = layer(out,out,out,label_embed,mask, att_heat_map)
        # CODE: sent level encoder end

        # # print('sent out.shape',out.shape)
        # # out - N,par_len,embed_size
        # # word_level_output - N,par_len, seq_len, embed_size - Basically for each element in the batch, 
        # # for each sentence in the abstract, we have a embed_size vector for each word
        # return out, word_level_outputs
        #CODE: sent level encoder return
        # return out


