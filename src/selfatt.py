import torch
import torch.nn as nn
import os

class SelfAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super(SelfAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads

        self.head_dim = embed_size // heads

        self.values = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.keys = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.queries = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.fc_out = nn.Linear(heads * self.head_dim, embed_size)

    def forward(self, values, keys, query, mask, att_heat_map = False):
        # Get number of training examples
        #Inputs - N,seq_len,embed_size
        N = query.shape[0]
        value_len, key_len, query_len = values.shape[1], keys.shape[1], query.shape[1]

        # Split the embedding into self.heads different pieces

        values = values.reshape(N, value_len, self.heads, self.head_dim)
        keys = keys.reshape(N, key_len, self.heads, self.head_dim)
        query = query.reshape(N, query_len, self.heads, self.head_dim)
        
        values = self.values(values) 
        keys = self.keys(keys) 
        queries = self.queries(query)
        if att_heat_map:
            if os.path.exists('batch_sent_att_vectors.pt'):
                b = list(torch.load('batch_sent_att_vectors.pt'))
            else:
                b = []        
            temp_query = queries.reshape(N,query_len,self.heads*self.head_dim)
            temp_key = keys.reshape(N,key_len,self.heads*self.head_dim)
            batch_scores = []
            for batch in range(temp_query.shape[0]):
                temp_q = temp_query[batch]          
                temp_k = temp_key[batch]            
                dot_att = torch.matmul(temp_k, temp_q.t())
                batch_scores.append(dot_att)
            batch_scores = torch.stack(batch_scores,dim=0)
            b.append(batch_scores)
            b = torch.stack(b,dim=0)
            torch.save(b,'batch_sent_att_vectors.pt')
            
        # Einsum does matrix mult. for query*keys for each training example
        # with every other training example, don't be confused by einsum
        # it's just how I like doing matrix multiplication & bmm

        energy = torch.einsum("nqhd,nkhd->nhqk", [queries, keys])

        # Mask padded indices so their weights become 0
        if mask is not None:
            energy = energy.masked_fill(mask == 0, float("-1e20"))

        # Normalize energy values similarly to seq2seq + attention
        # so that they sum to 1. Also divide by scaling factor for
        # better stability
        attention = torch.softmax(energy / (self.embed_size ** (1 / 2)), dim=3)

        out = torch.einsum("nhql,nlhd->nqhd", [attention, values]).reshape(
            N, query_len, self.heads * self.head_dim
        )
        # we reshape and flatten the last two dimensions.
        out = self.fc_out(out)
        return out