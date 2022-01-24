import torch
import torch.nn as nn
import argparse
from tqdm import tqdm
from torch.utils.data import DataLoader
from src.dataset import load_data, tokenize_and_pad, return_dataloader
from transformers import AutoTokenizer
from src.transformer import Transformer
import os
from sklearn.metrics import f1_score

PUBMED_LABEL_LIST = ['background','objective','methods','results','conclusions']   #pubmed
NICTA_LABEL_LIST = ['background','intervention','study design','population','outcome','other']  #nicta-pibosa
CSABSTRACT_LABEL_LIST = ['background','method','result','objective','other']    #cs-abstract

def convert_crf_output_to_tensor(output, max_par_len):
    out = []
    for o in output:
        if(len(o) < max_par_len):
            pad = [0 for _ in range(max_par_len - len(o))]
            o.extend(pad)
        out.append(o[:max_par_len])
    return torch.Tensor(out)
        
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=32)   
    parser.add_argument('--num_epochs',type=int,default=50)
    parser.add_argument('--lr',type=float,default=1e-3)
    parser.add_argument('--max_par_len',type=int,default=20)    
    parser.add_argument('--max_seq_len',type=int,default=128)  
    parser.add_argument('--dataset_name',type=str,default='pubmed', choices=['pubmed', 'nicta','csabstract'])
    parser.add_argument('--train_data',type=str,default='data/nicta_piboso/train_clean.txt')
    parser.add_argument('--dev_data',type=str,default='data/nicta_piboso/dev_clean.txt')
    parser.add_argument('--test_data',type=str,default='data/nicta_piboso/test_clean.txt')
    parser.add_argument('--bert_model',type=str,default="allenai/scibert_scivocab_uncased")
    parser.add_argument('--embed_size',type=int,default=120)
    parser.add_argument('--forward_expansion',type=int,default=4)
    parser.add_argument('--num_layers',type=int,default=3)
    parser.add_argument('--device',type=str,default='cuda')
    parser.add_argument('--save_model',type=bool,default=False)
    parser.add_argument('--save_path',type=str,default='models/')
    parser.add_argument('--load_model',type=bool,default=False)
    parser.add_argument('--load_path',type=str,default='models/')
    parser.add_argument('--seed',type=int,default=1234)
    parser.add_argument('--test_interval',type=int,default=1)
    args = parser.parse_args()
    print("Training arguments:")
    print(args)
    return args

def train(args):
    if args.device == 'cuda' and torch.cuda.is_available():
        device = torch.device('cuda')
        print("using gpu: ", torch.cuda.get_device_name(torch.cuda.current_device()))
        
    else:
        device = torch.device('cpu')
        print('using cpu')
    
    if args.dataset_name == 'pubmed':
        LABEL_LIST = PUBMED_LABEL_LIST
    elif args.dataset_name == 'nicta':
        LABEL_LIST = NICTA_LABEL_LIST
    elif args.dataset_name == 'csabstract':
        LABEL_LIST = CSABSTRACT_LABEL_LIST

    train_x,train_labels = load_data(args.train_data, args.max_par_len,LABEL_LIST)
    dev_x,dev_labels = load_data(args.dev_data, args.max_par_len,LABEL_LIST)
    test_x,test_labels = load_data(args.test_data, args.max_par_len,LABEL_LIST)

    tokenizer = AutoTokenizer.from_pretrained(args.bert_model)
    train_x = tokenize_and_pad(train_x,tokenizer,args.max_par_len,args.max_seq_len, LABEL_LIST)  ## N, par_len, seq_len
    dev_x = tokenize_and_pad(dev_x,tokenizer,args.max_par_len, args.max_seq_len, LABEL_LIST)
    test_x = tokenize_and_pad(test_x,tokenizer, args.max_par_len, args.max_seq_len, LABEL_LIST)

    training_params = {
        "batch_size": args.batch_size,
        "shuffle": True,
        "drop_last": False
        }
    dev_params = {
        "batch_size": args.batch_size,
        "shuffle": False,
        "drop_last": False
        }
    test_params = {
        "batch_size": args.batch_size,
        "shuffle": False,
        "drop_last": False
        }

    print('train.py train_x.shape:',train_x.shape,'train_labels.shape',train_labels.shape)
    training_generator = return_dataloader(inputs=train_x, labels=train_labels, params=training_params)
    dev_generator = return_dataloader(inputs=dev_x, labels=dev_labels, params=dev_params)
    test_generator = return_dataloader(inputs=test_x, labels=test_labels, params=test_params)   

    src_pad_idx = 0
    trg_pad_idx = 0
    model = Transformer(
        label_list=LABEL_LIST,
        src_pad_idx=src_pad_idx,
        trg_pad_idx=trg_pad_idx,
        embed_size=args.embed_size,
        num_layers=args.num_layers,   ## debug
        forward_expansion=args.forward_expansion,
        heads=len(LABEL_LIST),
        dropout=0.1,
        device=device,
        max_par_len=args.max_par_len,
        max_seq_len=args.max_seq_len,
        bert_model=args.bert_model
    )
    model = model.to(device).float()
    
    criterion = nn.CrossEntropyLoss(ignore_index=trg_pad_idx)
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
    
    epoch_losses = []
    best_val_loss = float('inf')
    for epoch in range(args.num_epochs):
        model.train()
        print(f"----------------[Epoch {epoch} / {args.num_epochs}]-----------------------")

        losses = []
        for batch_idx,batch in tqdm(enumerate(training_generator)):
            inp_data,target = batch
            inp_data = inp_data.to(device)
            target = target.to(device)

            ## For CRF
            optimizer.zero_grad()

            loss = -model(inp_data.long(),target[:,1:], training=True)       ## directly gives loss when training = True


            losses.append(loss.item())

            loss.backward()

            optimizer.step()
            
        mean_loss = sum(losses)/len(losses)

        print(f"Mean loss for epoch {epoch} is {mean_loss}")
        # Validation
        model.eval()
        val_targets = []
        val_preds = []
        for batch_idx,batch in tqdm(enumerate(dev_generator)):
            inp_data,target = batch
            inp_data = inp_data.to(device)
            target = target.to(device)
            with torch.no_grad():
                output = model(inp_data,target[:,:-1], training=False)      ## directly we get the labels here, instead of logits

            flattened_target = target[:,1:].to('cpu').flatten()
            output = convert_crf_output_to_tensor(output,args.max_par_len)
            flattened_preds = output.to('cpu').flatten()
            for target_i,pred_i in zip(flattened_target,flattened_preds):
                if target_i != 0:
                    val_targets.append(target_i)
                    val_preds.append(pred_i)

        f1 = f1_score(val_targets,val_preds,average='micro')
        
        print(f'------Micro F1 score on dev set: {f1}------')

        if loss < best_val_loss:
            print(f"val loss less than previous best val loss of {best_val_loss}")
            best_val_loss = loss
            if args.save_model:
                dir_name = f"seed_{args.seed}_parlen_{args.max_par_len}_seqlen_{args.max_seq_len}_lr_{args.lr}.pt"
                output_path = os.path.join(args.save_path,dir_name)
                if not os.path.exists(args.save_path):
                    os.makedirs(args.save_path)
                print(f"Saving model to path {output_path}")
                torch.save(model,output_path)

        # Testing
        if epoch % args.test_interval == 0:
            model.eval()
            test_targets = []
            test_preds = []
            for batch_idx, batch in tqdm(enumerate(test_generator)):
                inp_data,target = batch
                inp_data = inp_data.to(device)
                target = target.to(device)
                with torch.no_grad():
                    output = model(inp_data,target[:,:-1],training=False)
                    
                flattened_target = target[:,1:].to('cpu').flatten()
                output = convert_crf_output_to_tensor(output,args.max_par_len)
                flattened_preds = output.to('cpu').flatten()
                for target_i,pred_i in zip(flattened_target,flattened_preds):
                    if target_i!=0:
                        test_targets.append(target_i)
                        test_preds.append(pred_i)
            
            f1 = f1_score(test_targets,test_preds,average='micro')
            print(f"------Micro F1 score on test set: {f1}------")

    ## Uncomment for generating attention vectors. 
    ## Look into src/word_level_labelatt.py for details of computing and storing these attention scores
    ## Look into src/selfatt.py for sentence level attention scores
    # att_x = train_x[:10,:,:].to(device)
    # att_y = train_labels[:10,:].to(device)[:,:-1] 
    # model(att_x,att_y,training=False,att_heat_map=True)    

    ## Uncomment for getting error predictions
    # EXT_LABEL_LIST = ['<PAD>','<SOS>']+LABEL_LIST
    # with open("Errors.txt",'w',encoding='utf-8') as f:        
    #     for batch_idx,batch in tqdm(enumerate(test_generator)):
    #         inp_data,target = batch
    #         inp_data = inp_data.to(device)
    #         target = target.to(device)
    #         with torch.no_grad():
    #             output = model(inp_data,target[:,:-1],training=False)

    #         flattened_target = target[:,1:].to('cpu').flatten()
    #         output = convert_crf_output_to_tensor(output,args.max_par_len)
    #         flattened_preds = output.to('cpu').flatten()

    #         flattened_inp = inp_data.reshape(inp_data.shape[0]*inp_data.shape[1], -1)
    #         for target_i,pred_i,inp_i in zip(flattened_target,flattened_preds,flattened_inp):
    #             if target_i!=0 and (target_i != pred_i):
    #                 f.write(f"Target label: {EXT_LABEL_LIST[target_i.int()]}, Predicted label: {EXT_LABEL_LIST[pred_i.int()]}\n")
    #                 f.write(tokenizer.decode(inp_i,skip_special_tokens=True))
    #                 f.write("\n\n")


            


if __name__ == "__main__":

    args = get_args()
    train(args)