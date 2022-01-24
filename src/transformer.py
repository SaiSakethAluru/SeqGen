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
            heads,
            device,
            forward_expansion,
            dropout,
            max_par_len,
            max_seq_len,
        )
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
        src_mask = self.make_src_mask(src)
        trg_mask = self.make_crf_trg_mask(trg)
        enc_out = self.encoder(src,src_mask, att_heat_map)
        
        out = self.fc_out(enc_out)

        if training:
            crf_out = self.crf(out, trg, trg_mask, reduction='token_mean')

        else:
            crf_out = self.crf.decode(out,trg_mask)
    
        return crf_out
