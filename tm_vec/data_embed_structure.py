import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
import pickle
import gzip


#Define collate function for batches
def collate_fn(batch, pad_id = 1, max_len = 1024):
    S1_tensors = [t['Embed_sequence_1'] for t in batch]
    S2_tensors = [t['Embed_sequence_2'] for t in batch]
    combined_tensors = S1_tensors + S2_tensors

    padded_tensors = torch.nn.utils.rnn.pad_sequence(combined_tensors, padding_value=pad_id, batch_first=True)
    padded_tensors = padded_tensors[:, :max_len, :]

    batch_length = len(batch)
    S1_tensors_padded = padded_tensors[:batch_length]
    S2_tensors_padded = padded_tensors[batch_length:]
    
    pad_labels_seq1 = torch.zeros(S1_tensors_padded.shape[0:2]).type(torch.BoolTensor)
    pad_labels_seq2 = torch.zeros(S2_tensors_padded.shape[0:2]).type(torch.BoolTensor)
    pad_labels_seq1[S1_tensors_padded[:,:,0] == pad_id] = True  #pad_id 
    pad_labels_seq2[S2_tensors_padded[:,:,0] == pad_id] = True  #pad_id 

    tm_scores = torch.tensor([t['tm_score'] for t in batch])
    
    return(S1_tensors_padded, S2_tensors_padded, pad_labels_seq1, pad_labels_seq2, tm_scores)


class tm_score_embeds_dataset(Dataset):
    """TM score dataset."""

    def __init__(self, pickle_file):
        """
        Args:
            pickle_file (string): Path to the pickle file with data.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.tm_score_emb_data = pickle.load(open(pickle_file, "rb"))
        
        #with gzip.open(pickle_file, "rb") as f:
        #    self.tm_score_emb_data = pickle.load(f) 

    def __len__(self):
        return len(self.tm_score_emb_data)

    def __getitem__(self, idx):
        data_sample = self.tm_score_emb_data[idx]
        seq1 = data_sample['Embed_sequence_1']
        seq2 = data_sample['Embed_sequence_2']
        tm_score = data_sample['tm_score']
        sample = {'Embed_sequence_1': seq1, 'Embed_sequence_2': seq2, 'tm_score': tm_score}
        return sample


#Function to construct datasets to be fed to dataloader
def construct_datasets(data_dir, train_prop=.9, val_prop=.05, test_prop = .05):
    total_samples = len(tm_score_embeds_dataset(data_dir))
    sampleable_values = np.arange(total_samples)

    train_n_to_sample = int(len(sampleable_values) * train_prop)
    val_n_to_sample = int(len(sampleable_values) * val_prop)
    test_n_to_sample = int(len(sampleable_values) * test_prop)

    train_indices = np.random.choice(sampleable_values, train_n_to_sample, replace=False)
    sampleable_values = sampleable_values[~np.isin(sampleable_values, train_indices)]
    val_indices = np.random.choice(sampleable_values, val_n_to_sample, replace=False)
    sampleable_values = sampleable_values[~np.isin(sampleable_values, val_indices)]
    test_indices = np.random.choice(sampleable_values, test_n_to_sample, replace=False)

    #Make train, test, and validation datasets using torch subset
    train_ds = torch.utils.data.Subset(tm_score_embeds_dataset(data_dir), train_indices)
    val_ds =  torch.utils.data.Subset(tm_score_embeds_dataset(data_dir), val_indices)
    test_ds = torch.utils.data.Subset(tm_score_embeds_dataset(data_dir), test_indices)
    
    return(train_ds, val_ds, test_ds)



