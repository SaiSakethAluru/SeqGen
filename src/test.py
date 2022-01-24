import torch
import torch.nn as nn
from src.transformer import Transformer

src_vocab_size = 10
embed_size = 8
num_layers = 5
heads = 4
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
forward_expansion = 4
dropout = 0.8
max_length = 10
labels = torch.tensor([1,2,3,4]).to(device)
batch_size = 2
par_len = 3
src_pad_idx = 0
trg_pad_idx = 0
model = Transformer(
    src_vocab_size,
    len(labels),
    src_pad_idx,
    trg_pad_idx,
    labels,
    embed_size,
    num_layers,
    forward_expansion,
    heads,
    dropout,
    device,
    max_length
)

x = torch.randint(0,10,size=(batch_size,par_len,max_length)).to(device)
trg = torch.randint(0,len(labels),size=(batch_size,par_len)).to(device)
output = model(x,trg)

print(output)
