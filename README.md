[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/mma-dfer-multimodal-adaptation-of-unimodal/dynamic-facial-expression-recognition-on-dfew)](https://paperswithcode.com/sota/dynamic-facial-expression-recognition-on-dfew?p=mma-dfer-multimodal-adaptation-of-unimodal)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/mma-dfer-multimodal-adaptation-of-unimodal/dynamic-facial-expression-recognition-on-mafw)](https://paperswithcode.com/sota/dynamic-facial-expression-recognition-on-mafw?p=mma-dfer-multimodal-adaptation-of-unimodal)

# MMA-DFER: MultiModal Adaptation of unimodal models for Dynamic Facial Expression Recognition in-the-wild

This repository provides an official implementation for the paper [MMA-DFER: MultiModal Adaptation of unimodal models for Dynamic Facial Expression Recognition in-the-wild]( 
https://arxiv.org/abs/2404.09010).

![a](https://github.com/katerynaCh/av-emotion-recognition-in-the-wild/blob/main/fff.drawio.png)

## Installation

Please create an environment with Python 3.10 and use requirements file to install the rest of the libraries

```bash
pip install -r reqiurements.txt
```

## Data preparation

We provide the codes for [DFEW](https://dfew-dataset.github.io/) and [MAFW](https://mafw-database.github.io/MAFW/) datasets, which you would need to download. The annotations are provided in annotations/ directory. You would need to update the paths to your own - there are preprocessing scripts in the same directory to preprocess data and rename the paths. 

For MAFW dataset, you would need to extract faces from videos. Please refer to data_utils that has an example of face detection pipeline. 
To extract audio from the video files (in both datasets), use the [following script](https://github.com/katerynaCh/MMA-DFER/blob/main/mp42wav.sh) (after modifying the paths to your own).

You will also need to download pre-trained checkpoints for vision encoder from [https://github.com/FuxiVirtualHuman/MAE-Face](https://github.com/FuxiVirtualHuman/MAE-Face/releases) and for audio encoder from [https://github.com/facebookresearch/AudioMAE](https://drive.usercontent.google.com/download?id=1ni_DV4dRf7GxM8k-Eirx71WP9Gg89wwu&export=download&authuser=0) Please extract them and rename the audio checkpoint to 'audiomae_pretrained.pth'. Both checkpoints are expected to be in root folder.

## Running the code

The main script in main.py. You can invoke it through running:
```bash
./train_DFEW.sh
```
```bash
./train_MAFW.sh
```

## Evaluation

You can download pre-trained models on DFEW from [here](https://drive.google.com/drive/folders/1I3dvClr4oVH3h5cGmaFXmTc8aV-nYHof?usp=sharing) and on MAFW from [here](https://drive.google.com/drive/folders/1I5baHCpBdU5RT48yaFyNhnZoQkbHm-OS?usp=sharing). **Please respect the dataset license when downloading the models!** Evaluation can be done as follows:
```bash
python evaluate.py --fold $FOLD --checkpoint $CHECKPOINT_PATH --img-size $IMG_SIZE --dataset [MAFW|DFEW]
```

## References
This repository is based on DFER-CLIP https://github.com/zengqunzhao/DFER-CLIP. We also thank the authors of MAE-Face https://github.com/FuxiVirtualHuman/MAE-Face and Audiomae https://github.com/facebookresearch/AudioMAE

## Citation
If you use our work, please cite as:
```
@InProceedings{Chumachenko_2024_CVPR,
    author    = {Chumachenko, Kateryna and Iosifidis, Alexandros and Gabbouj, Moncef},
    title     = {MMA-DFER: MultiModal Adaptation of Unimodal Models for Dynamic Facial Expression Recognition In-the-wild},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) Workshops},
    month     = {June},
    year      = {2024},
    pages     = {4673-4682}
}
```
