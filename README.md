# MMA-DFER: MultiModal Adaptation of unimodal models for Dynamic Facial Expression Recognition in-the-wild

This repository provides an official implementation for the paper [MMA-DFER: MultiModal Adaptation of unimodal models for Dynamic Facial Expression Recognition in-the-wild]( 
https://arxiv.org/abs/2404.09010).

## Installation

Please create an environment with Python 3.10 and use requirements file to install the rest of the libraries

```bash
pip install -r reqiurements.txt
```

## Data preparation

We provide the codes for [DFEW](https://dfew-dataset.github.io/) and [MAFW](https://mafw-database.github.io/MAFW/) datasets, which you would need to download. Then, please refer to DFER-CLIP repository for transforming the annotations that are provided in annotations/ folder. To extract faces from MAFW dataset, please refer to data_utils.

## Running the code

The main script in main.py. You can invoke it through running:
```bash
./train_DFEW.sh
```
```bash
./train_MAFW.sh
```

## References
This repository is based on DFER-CLIP https://github.com/zengqunzhao/DFER-CLIP. We also thank the authors of MAE-Face https://github.com/FuxiVirtualHuman/MAE-Face and Audiomae https://github.com/facebookresearch/AudioMAE

## Citation
If you use our work, please cite as:

@article{chumachenko2024mma,
  title={MMA-DFER: MultiModal Adaptation of unimodal models for Dynamic Facial Expression Recognition in-the-wild},
  author={Chumachenko, Kateryna and Iosifidis, Alexandros and Gabbouj, Moncef},
  journal={arXiv preprint arXiv:2404.09010},
  year={2024}
}
