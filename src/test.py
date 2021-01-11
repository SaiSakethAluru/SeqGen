import torch
import torch.nn as nn
# from sent_encoder import SentenceEncoder
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
# model = SentenceEncoder(src_vocab_size,embed_size,num_layers,heads,device,forward_expansion,dropout,max_length,labels)
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

# x = torch.tensor([[[1,5,6,3,2,4,2,4,1,0],[1,3,2,4,3,2,5,0,0,0],]]).to(device)
x = torch.randint(0,10,size=(batch_size,par_len,max_length)).to(device)
# trg = torch.tensor([1]).expand(batch_size,-1).to(device)
trg = torch.randint(0,len(labels),size=(batch_size,par_len)).to(device)
# pad_idx = 0
# mask = (x != pad_idx).to(device)
print(x.shape)
output = model(x,trg)

print(output)
# print(word_lev_outputs)
print("output.shape",output.shape)
# print("word_lev_outputs.shape", word_lev_outputs.shape)
