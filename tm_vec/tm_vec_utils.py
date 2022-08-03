import numpy as np
import pandas as pd

import torch
from torch import nn
from tm_vec.embed_structure_model import trans_basic_block, trans_basic_block_Config

from transformers import T5EncoderModel, T5Tokenizer
import re


#Function to extract ProtTrans embedding for a sequence
def featurize_prottrans(sequences, model, tokenizer, device): 
    
    sequences = [(" ".join(sequences[i])) for i in range(len(sequences))]
    sequences = [re.sub(r"[UZOB]", "X", sequence) for sequence in sequences]
    ids = tokenizer.batch_encode_plus(sequences, add_special_tokens=True, padding=True)
    input_ids = torch.tensor(ids['input_ids']).to(device)
    attention_mask = torch.tensor(ids['attention_mask']).to(device)

    with torch.no_grad():
        embedding = model(input_ids=input_ids, attention_mask=attention_mask)

    embedding = embedding.last_hidden_state.cpu().numpy()

    features = [] 
    for seq_num in range(len(embedding)):
        seq_len = (attention_mask[seq_num] == 1).sum()
        seq_emd = embedding[seq_num][:seq_len-1]
        features.append(seq_emd)
    
    prottrans_embedding = torch.tensor(features[0])
    prottrans_embedding = torch.unsqueeze(prottrans_embedding, 0).to(device)
    
    return(prottrans_embedding)



#Embed a protein using tm_vec (takes as input a prottrans embedding)
def embed_tm_vec(prottrans_embedding, model_deep, device):
    padding = torch.zeros(prottrans_embedding.shape[0:2]).type(torch.BoolTensor).to(device)
    tm_vec_embedding = model_deep(prottrans_embedding, src_mask=None, src_key_padding_mask=padding)

    return(tm_vec_embedding.cpu().detach().numpy())


#Predict the TM-score for a pair of proteins (inputs are TM-Vec embeddings)
def cosine_similarity_tm(output_seq1, output_seq2):
    cos = nn.CosineSimilarity(dim=1, eps=1e-6)
    dist_seq = cos(output_seq1, output_seq2)

    return(dist_seq)
