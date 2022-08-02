#!/usr/bin/env sh

python ../scripts/tm_vec_run.py \
    --input_data /mnt/home/thamamsy/ceph/cath/cath/cath_sequence_data/cath_sequence_data/cath-domain-seqs-S100.fa \
    --database /mnt/home/thamamsy/ceph/deepblast/data/embeddings_cath_s100_final.npy \
    --tm_vec_model_path /mnt/home/thamamsy/ceph/deepblast/models/transformer_lr0.0001_dmodel1024_nlayer2_datasample_45_thresh_300_pairs_in_folds_included_23M_normal_tmax/checkpoints/last.ckpt \
    --tm_vec_config_path /mnt/home/thamamsy/ceph/deepblast/models/transformer_lr0.0001_dmodel1024_nlayer2_datasample_45_thresh_300_pairs_in_folds_included_23M_normal_tmax/params.json \
    --metadata /mnt/home/thamamsy/ceph/deepblast/data/embeddings_cath_s100_ids.npy \
    --path_output_neigbhors /mnt/home/thamamsy/projects/deep_blast_ir/deep_blast_embed_dev/tm_vec_all/tmp \
    --path_output_embeddings /mnt/home/thamamsy/projects/deep_blast_ir/deep_blast_embed_dev/tm_vec_all/tmp2 \
    --align True --tm_vec_align_path /mnt/home/thamamsy/projects/deepblast/deepblast-lstm4x.pt \
    --database_sequences /mnt/home/thamamsy/ceph/cath/cath/cath_sequence_data/cath_sequence_data/cath-domain-seqs-S100.fa \
    --path_output_alignments /mnt/home/thamamsy/projects/deep_blast_ir/deep_blast_embed_dev/tm_vec_all/tmp3

#Required inputs
#"--input_data": These are the sequences you want to query (FASTA format)
#"--database": This is the database of vectors to query (i.e. CATH, SwissProt)
#"--tm_vec_model_path": TM-Vec Model path for embedding
#"--tm_vec_config_path": Config for TM-Vec Model
#"--k_nearest_neighbors": Number of nearest neighbhors
#"--path_output_embeddings": Path for saving Embeddings for queried sequences
#"--path_output_neigbhors": Path for outputting the nearest neighbors (either their indices or metadata if its supplied)

#Optional inputs
#"--metadata": Metadata for queried database (will return this metadata instead of indices if this is supplied)

#If performing alignments:
#"--align": True of False- would you like to also perform alignments for query results
#"--tm_vec_align_path": TM-Vec-Align model path
#"--database_sequences": Database of sequences that corresponds to the query database of embedding vectors
#"--path_output_alignments": Path for saving alignments
