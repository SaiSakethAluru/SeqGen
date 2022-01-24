import torch
import torch.nn as nn



class WordLabelAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super(WordLabelAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads

        self.values = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.keys = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.queries = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.fc_out = nn.Linear(heads * self.head_dim, embed_size)

    def forward(self, values, keys, query, mask, att_heat_map=False):
        # Get number of training examples
        N = query.shape[0]
        value_len, key_len, query_len = values.shape[1], keys.shape[1], query.shape[1]
        query_seq_len = query.shape[2]

        # Split the embedding into self.heads different pieces

        values = values.reshape(N, value_len, self.heads, self.head_dim)
        keys = keys.reshape(N, key_len, self.heads, self.head_dim)
        query = query.reshape(N, query_len, query_seq_len, self.heads, self.head_dim)

        values = self.values(values) 
        keys = self.keys(keys)  
        queries = self.queries(query)

        if att_heat_map:
            temp_query = queries.reshape(N,query_len, query_seq_len, self.heads*self.head_dim)
            temp_key = keys.reshape(N,key_len,self.heads*self.head_dim)
            batch_scores = []
            for b in range(temp_query.shape[0]):
                att_scores = []
                for i in range(temp_query.shape[1]):
                    temp_q = temp_query[b][i]      
                    temp_k = temp_key[b]            
                    dot_att = torch.matmul(temp_k, temp_q.t())
                    att_scores.append(dot_att)
                att_scores = torch.stack(att_scores,dim=0)
                batch_scores.append(att_scores)
            batch_scores = torch.stack(batch_scores,dim=0)
            torch.save(batch_scores,'batch_att_vectors.pt')    

        # Einsum does matrix mult. for query*keys for each training example
        # with every other training example, don't be confused by einsum
        # it's just how I like doing matrix multiplication & bmm

        energy = torch.einsum("nqshd,nkhd->nhqks", [queries, keys])

        # Mask padded indices so their weights become 0
        if mask is not None:
            energy = energy.masked_fill(mask == 0, float("-1e20"))

        # Normalize energy values similarly to seq2seq + attention
        # so that they sum to 1. Also divide by scaling factor for
        # better stability
        attention = torch.softmax(energy / (self.embed_size ** (1 / 2)), dim=3)

        out = torch.einsum("nhqls,nlhd->nqshd", [attention, values]).reshape(
            N, query_len, query_seq_len, self.heads * self.head_dim
        )
        # we reshape and flatten the last two dimensions.
        out = self.fc_out(out)
        return out