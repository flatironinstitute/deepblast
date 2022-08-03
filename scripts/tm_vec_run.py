#!/usr/bin/env python3
import numpy as np
import pandas as pd
import torch
from tm_vec.embed_structure_model import trans_basic_block, trans_basic_block_Config
from tm_vec.tm_vec_utils import featurize_prottrans, embed_tm_vec
from transformers import T5EncoderModel, T5Tokenizer
import re
import gc
from Bio import SeqIO
import faiss
from pathlib import Path
import argparse
import pickle
from tm_vec_align.trainer import LightningAligner
from tm_vec_align.dataset.utils import pack_sequences
from tm_vec_align.dataset.utils import states2alignment

parser = argparse.ArgumentParser(description='Process TM-Vec arguments')

parser.add_argument("--input_data",
        type=Path,
        required=True,
        help="Input data"
)

parser.add_argument("--database",
        type=Path,
        required=True,
        help="Database to query"
)


parser.add_argument("--metadata",
        type=Path,
        help="Metadata for queried database"
)

parser.add_argument("--tm_vec_model_path",
        type=Path,
        required=True,
        help="Model path for embedding"
)

parser.add_argument("--tm_vec_config_path",
        type=Path,
        required=True,
        help="Config path for embedding"
)

parser.add_argument("--tm_vec_align_path",
        type=Path,
        help="Align model path"
)

parser.add_argument("--k_nearest_neighbors",
        type=int,
        default=5,
        help="Number of nearest neighbhors"
)


parser.add_argument("--align",
        type=bool,
        help="Whether to return alignments"
)

parser.add_argument("--database_sequences",
        type=Path,
        help="Database sequences"
)


parser.add_argument("--path_output_neigbhors",
        type=Path,
        required=True,
        help="Nearest neighbor outputs"
)

parser.add_argument("--path_output_embeddings",
        type=Path,
        help="Embeddings for queried sequences"
)

parser.add_argument("--path_output_alignments",
        type=Path,
        help="Alignment output file if alignment is true"
)

#Load arguments
args = parser.parse_args()

#Set device
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

#Load the ProtTrans model and ProtTrans tokenizer
tokenizer = T5Tokenizer.from_pretrained("Rostlab/prot_t5_xl_uniref50", do_lower_case=False )
model = T5EncoderModel.from_pretrained("Rostlab/prot_t5_xl_uniref50")
gc.collect()
model = model.to(device)
model = model.eval()
print("ProtTrans model downloaded")


#Load the Tm_Vec_Align TM model
tm_vec_model_config = trans_basic_block_Config.from_json(args.tm_vec_config_path)
model_deep = trans_basic_block.load_from_checkpoint(args.tm_vec_model_path, config=tm_vec_model_config)
model_deep = model_deep.to(device)
model_deep = model_deep.eval()
print("TM-Vec model loaded")


#Read in query sequences
with open(args.input_data) as handle:
    headers = []
    seqs = []
    for record in SeqIO.parse(handle, "fasta"):
        headers.append(record.id)
        seqs.append(str(record.seq))

flat_seqs = [seqs[i] for i in range(len(seqs))]
print("Sequences inputed")

#Embed all query sequences
i = 0
embed_all_sequences= []
while i < len(flat_seqs):
    seq = flat_seqs[i:i+1]
    protrans_sequence = featurize_prottrans(seq, model, tokenizer, device)
    embed_sequence = embed_tm_vec(protrans_sequence, model_deep, device)
    embed_all_sequences.append(embed_sequence)
    i = i + 1

#Read in query database
query_database = np.load(args.database)

#Build query array and save the embeddings (will write them to output)
embed_all_sequences_np = np.concatenate(embed_all_sequences, axis=0)
queries = embed_all_sequences_np.copy()
#Normalize queries
faiss.normalize_L2(queries)

#Build an indexed database
d = query_database.shape[1]
index = faiss.IndexFlatIP(d)
faiss.normalize_L2(query_database)
index.add(query_database)

#Return the nearest neighbors
k = args.k_nearest_neighbors
D, I = index.search(queries, k)


#Load metadata if it exists
if args.metadata is not None:
    metadata_database = np.load(args.metadata)
    #Return the metadata for the nearest neighbor results
    near_ids = []

    print("meta data loaded")

    for i in range(I.shape[0]):
        meta = metadata_database[I[i]]
        near_ids.append(list(meta))

    near_ids = np.array(near_ids)


#Alignment section
#If we are also aligning proteins, load tm_vec align and align nearest neighbors if true
if args.align == True:
    align_model = LightningAligner.load_from_checkpoint(args.tm_vec_align_path)
    align_model = align_model.to(device)

    #Read in database sequences
    with open(args.database_sequences) as handle_db:
        seqs_db = []
        for record in SeqIO.parse(handle_db, "fasta"):
            seqs_db.append(str(record.seq))

    alignments = []
    for i in range(I.shape[0]):
        alignments_i = []
        for j in range(I.shape[1]):
            x = flat_seqs[i]
            y = seqs_db[I[i,j]]
            pred_alignment = align_model.align(x, y)
            x_aligned, y_aligned = states2alignment(pred_alignment, x, y)
            alignments_i.append([x_aligned, pred_alignment, y_aligned])

        alignments.append(alignments_i)

    #Write out the alignments
    np.save(args.path_output_alignments, np.array(alignments))


#Outputting results
#Write out the nearest neighbors (if no database meta data is provided, write out their database indices)
if args.metadata is not None:
    np.save(args.path_output_neigbhors, near_ids)
else:
    np.save(args.path_output_neigbhors, I)


#Write out the embeddings
np.save(args.path_output_embeddings, embed_all_sequences_np)

print("Done")
