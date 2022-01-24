import torch
import torch.nn as nn


class SelfAttention3D(nn.Module):
    def __init__(self, embed_size, heads):
        super(SelfAttention3D, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size


        self.values = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.keys = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.queries = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.fc_out = nn.Linear(heads * self.head_dim, embed_size)

    def forward(self, values, keys, query, mask):
        # Get number of training examples
        #Inputs - N,seq_len,embed_size
        N = query.shape[0]
        value_len, key_len, query_len = values.shape[1], keys.shape[1], query.shape[1]
        value_seq_len, key_seq_len, query_seq_len = values.shape[2], keys.shape[2], query.shape[2]
        # Split the embedding into self.heads different pieces

        values = values.unsqueeze(3).expand(N,value_len,value_seq_len, self.heads,self.head_dim)
        keys = keys.unsqueeze(3).expand(N,key_len,key_seq_len, self.heads,self.head_dim)
        query = query.unsqueeze(3).expand(N,query_len,query_seq_len, self.heads,self.head_dim)

        values = self.values(values) 
        keys = self.keys(keys) 
        queries = self.queries(query)

        # Einsum does matrix mult. for query*keys for each training example
        # with every other training example, don't be confused by einsum
        # it's just how I like doing matrix multiplication & bmm
        energy = torch.einsum("nqshd,nkshd->nhqks", [queries, keys])
        # Mask padded indices so their weights become 0
        if mask is not None:
            energy = energy.masked_fill(mask == 0, float("-1e20"))

        # Normalize energy values similarly to seq2seq + attention
        # so that they sum to 1. Also divide by scaling factor for
        # better stability
        attention = torch.softmax(energy / (self.embed_size ** (1 / 2)), dim=3)

        out = torch.einsum("nhqls,nlshd->nqshd", [attention, values]).reshape(
            N, query_len, query_seq_len, self.heads * self.head_dim
        )
        out = self.fc_out(out)
        return out