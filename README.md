[![DOI](https://zenodo.org/badge/269478463.svg)](https://zenodo.org/badge/latestdoi/269478463)


# DeepBLAST 

Learning protein structural similarity from sequence alone.  Our preprint can be found [here](https://www.biorxiv.org/content/10.1101/2020.11.03.365932v1)

DeepBLAST is a neural-network based alignment algorithm that can estimate structural alignments. And it can generate structural alignments that are nearly identical to
state-of-the-art structural alignment algorithms.
![Malidup benchmark](/imgs/malidup.png)

# Installation

DeepBLAST can be installed from pip via

```
pip install deepblast
```

To install from the development branch run

```
pip install git+https://github.com/flatironinstitute/deepblast.git
```

# Downloading pretrained models and data

The pretrained DeepBLAST model can be downloaded [here](https://users.flatironinstitute.org/jmorton/public_www/deepblast-public-data/checkpoints/deepblast-l8.ckpt).

The TM-align structural alignments used to pretrain DeepBLAST can be found below
- [Training data](https://users.flatironinstitute.org/jmorton/public_www/deepblast-public-data/train_matched.txt)
- [Validation data](https://users.flatironinstitute.org/jmorton/public_www/deepblast-public-data/valid.txt)
- [Testing data](https://users.flatironinstitute.org/jmorton/public_www/deepblast-public-data/test.txt)


See the [Malisam](http://prodata.swmed.edu/malisam/) and [Malidup](http://prodata.swmed.edu/malidup/) websites to download their datasets.

# Getting started

See the [wiki](https://github.com/flatironinstitute/deepblast/wiki) on how to use DeepBLAST and TM-vec for remote homology search and alignment.
If you have questions on how to use DeepBLAST and TM-vec, feel free to raise questions in the [discussions section](https://github.com/flatironinstitute/deepblast/discussions). If you identify any potential bugs, feel free to raise them in the [issuetracker](https://github.com/flatironinstitute/deepblast/issues)

# Citation

If you find our work useful, please cite us at
```
@article{morton2020protein,
  title={Protein Structural Alignments From Sequence},
  author={Morton, Jamie and Strauss, Charlie and Blackwell, Robert and Berenberg, Daniel and Gligorijevic, Vladimir and Bonneau, Richard},
  journal={bioRxiv},
  year={2020},
  publisher={Cold Spring Harbor Laboratory}
}

@article{hamamsy2022tm,
  title={TM-Vec: template modeling vectors for fast homology detection and alignment},
  author={Hamamsy, Tymor and Morton, James T and Berenberg, Daniel and Carriero, Nicholas and Gligorijevic, Vladimir and Blackwell, Robert and Strauss, Charlie EM and Leman, Julia Koehler and Cho, Kyunghyun and Bonneau, Richard},
  journal={bioRxiv},
  pages={2022--07},
  year={2022},
  publisher={Cold Spring Harbor Laboratory}
}

```
