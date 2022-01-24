import transformers
import torch
from torch.utils.data import TensorDataset, DataLoader 
from nltk.tokenize import sent_tokenize



LABEL_ENCODE = None

def get_label_encodes(tokenizer, label_list):
    token_ids = []
    sep_token = 102
    for label in label_list:
        label_ids = tokenizer.encode(label)
        token_ids.extend(label_ids)
        token_ids.append(sep_token)
    return token_ids


def load_data(data_path, max_par_len, label_list):
    texts, labels = [], []
    abstract = ""
    abs_labels = [1]    ## Initial sos token 
    with open(data_path, encoding='utf-8') as data_file:
        data_file_lines = data_file.readlines()
        data_file_lines = list(filter(None,[line.rstrip() for line in data_file_lines]))
        for line in data_file_lines[1:]:  # ignore first line which is ID
            if line.startswith('###'):
                texts.append(abstract)
                if(len(abs_labels) < max_par_len+1):
                    pad_labels = [0 for _ in range(max_par_len+1 - len(abs_labels))]
                    abs_labels.extend(pad_labels)
                labels.append(abs_labels[:max_par_len+1])
                abstract = ""
                abs_labels = [1]
                continue
            label, txt = line.split('\t', 1)
            abstract += txt.lower()+'\n'
            abs_labels.append(label_list.index(label.lower())+2)
        
    return texts, torch.Tensor(labels).long()



def tokenize(paragraphs, tokenizer, max_par_len, max_seq_len, label_list):
    input_ids = []
    for para in paragraphs:
        sentences = sent_tokenize(para)
        para_ids = []
        for sent in sentences[:max_par_len]:
            try:
                encoded_sent = tokenizer.encode(
                    sent,
                    add_special_tokens = True,
                    max_length = max_seq_len
                )
            except ValueError: 
                encoded_sent = tokenizer.encode(
                    '',
                    add_special_tokens = True,
                    max_length = max_seq_len
                )
            
            if(len(encoded_sent) < max_seq_len):
                pad_words = [0 for _ in range(max_seq_len - len(encoded_sent))]
                encoded_sent.extend(pad_words)
            para_ids.append(encoded_sent[:max_seq_len])
        if(len(para_ids) < max_par_len):
            pad_sentences = [[0 for _ in range(max_seq_len)] for _ in range(max_par_len - len(para_ids))]
            para_ids.extend(pad_sentences)
        input_ids.append(para_ids[:max_par_len])
    return input_ids        ## Will be a list of len N. Each entry para_len x seq_len list.



def tokenize_and_pad(paragraphs, tokenizer, max_par_len, max_seq_len, label_list):
    input_ids = tokenize(paragraphs,tokenizer,max_par_len,max_seq_len, label_list)
    input_ids = torch.Tensor(input_ids).long()
    input_ids = input_ids[:,:max_par_len, :max_seq_len]        
    return input_ids

def return_dataloader(inputs, labels, params):
    data = TensorDataset(inputs, labels)
    dataloader = DataLoader(data, **params)
    return dataloader
