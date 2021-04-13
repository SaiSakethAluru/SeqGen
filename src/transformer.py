import torch
import torch.nn as nn
from src.sent_encoder import SentenceEncoder
from src.decoder import Decoder
from torchcrf import CRF

class Transformer(nn.Module):
    def __init__(
            self,
            label_list,
            src_pad_idx,
            trg_pad_idx,
            batch_size,
            embed_size=512,
            num_layers=6,
            forward_expansion=4,
            heads=8,
            dropout=0,
            device="cpu",
            max_par_len=10,
            max_seq_len=20,
            bert_model="allenai/scibert_scivocab_uncased",
    ):
        super(Transformer, self).__init__()
        self.encoder = SentenceEncoder(
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
        )
        # self.decoder = Decoder(
        #     label_list,
        #     embed_size,
        #     num_layers,
        #     heads,
        #     forward_expansion,
        #     dropout,
        #     device,
        #     max_par_len,
        #     max_seq_len,
        #     embed_path
        # )
        # self.second_pass_decoder = Decoder(
        #     label_list,
        #     embed_size,
        #     num_layers,
        #     heads,
        #     forward_expansion,
        #     dropout,
        #     device,
        #     max_par_len,
        #     max_seq_len,
        #     embed_path
        self.fc_out = nn.Linear(embed_size,len(label_list)+2)
        self.crf = CRF(len(label_list)+2, batch_first = True)
        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = trg_pad_idx
        self.device = device
    
    def make_src_mask(self,src):
        src_mask = (src != self.src_pad_idx).int()
        return src_mask.to(self.device)
    
    def make_trg_mask(self,trg):
        N,trg_len = trg.shape
        trg_mask = torch.tril(torch.ones((trg_len,trg_len))).expand(
            N,1,trg_len,trg_len
        )
        return trg_mask.to(self.device)

    def make_second_pass_mask(self,trg):
        N,trg_len = trg.shape
        trg_mask = torch.ones((trg_len,trg_len)).expand(
            N,1,trg_len,trg_len
        )
        return trg_mask.to(self.device)

    def make_crf_trg_mask(self, trg):
        trg_mask = (trg != self.trg_pad_idx)
        return trg_mask.to(self.device)

    def forward(self,src,trg, training, att_heat_map=False):
        # print('transformer src.shape',src.shape)
        # src --> N, par_len, seq_len -> 2,3,10
        # print('transformer trg.shape',trg.shape)
        # trg --> N,par_len -> 2,3
        src_mask = self.make_src_mask(src)
        # src_mask --> N,par_len, seq_len -> 2,3,10
        # trg_mask = self.make_trg_mask(trg)
        trg_mask = self.make_crf_trg_mask(trg)
        # trg_mask --> N, 1, par_len, par_len -> 2,1,3,3
        # print('transformer src_mask.shape',src_mask.shape)
        # print('transformer trg_mask.shape',trg_mask.shape)

        # enc_out,enc_word_out = self.encoder(src,src_mask) 
        enc_out = self.encoder(src,src_mask, att_heat_map)
        
        out = self.fc_out(enc_out)

        if training:
            crf_out = self.crf(out, trg, trg_mask, reduction='token_mean')

        else:
            # loss = self.crf(out,trg,trg_mask)     ## should this be here to get val loss? 
            crf_out = self.crf.decode(out,trg_mask)
    
        return crf_out

        # # enc_out --> N,par_len,embed_size -> 2,3,8
        # # enc_word_out --> N,par_len, seq_len, embed_size -> 2,3,10,8
        # # print("transformer enc_out.shape",enc_out.shape)
        # # print('transformer enc_word_out.shape',enc_word_out.shape)
        # N,par_len,seq_len = src.shape
        # word_mask = torch.eye(src.shape[1]).unsqueeze(2).cuda()
        # # word_mask --> par_len x par_len x 1
        # word_mask = word_mask.expand(src.shape[0],-1,-1,src.shape[2]).bool()
        # # word_mask --> N,par_len,par_len,seq_len
        # pad_mask = src_mask.unsqueeze(1).expand(-1,par_len,-1,-1).bool()
        # # pad_mask --> N,par_len,par_len,seq_len
        # final_word_mask = (word_mask & pad_mask).unsqueeze(1)
        # # print('transformer final_word_mask.shape',final_word_mask.shape)
        # # final_word_mask --> N,1,par_len,par_len,seq_len
        # out = self.decoder(trg,enc_out,enc_word_out,src_mask,final_word_mask,trg_mask)
        # # print('transformer out.shape',out.shape)
        # # out --> N,par_len,num_labels
        # labels = out.argmax(dim=-1)
        # second_pass_mask = self.make_second_pass_mask(trg)
        # second_out = self.second_pass_decoder(labels,enc_out,enc_word_out,src_mask,final_word_mask,second_pass_mask)
        # return second_out
