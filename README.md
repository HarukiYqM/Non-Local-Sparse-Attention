# Image Super-Resolution with Non-Local Sparse Attention 
This repository is for NLSN introduced in the following paper "Image Super-Resolution with Non-Local Sparse Attention", CVPR2021, [[Link]](https://openaccess.thecvf.com/content/CVPR2021/papers/Mei_Image_Super-Resolution_With_Non-Local_Sparse_Attention_CVPR_2021_paper.pdf) 


The code is built on [EDSR (PyTorch)](https://github.com/thstkdgus35/EDSR-PyTorch) and test on Ubuntu 18.04 environment (Python3.6, PyTorch > 1.1.0) with V100 GPUs. 
## Contents
1. [Introduction](#introduction)
2. [Train](#train)
3. [Test](#test)
5. [Citation](#citation)
6. [Acknowledgements](#acknowledgements)

## Introduction

Both Non-Local (NL) operation and sparse representa-tion are crucial for Single Image Super-Resolution (SISR).In this paper, we investigate their combinations and proposea novel Non-Local Sparse Attention (NLSA) with dynamicsparse attention pattern. NLSA is designed to retain long-range modeling capability from NL operation while enjoying robustness and high-efficiency of sparse representation.Specifically, NLSA rectifies non-local attention with spherical locality sensitive hashing (LSH) that partitions the input space into hash buckets of related features. For everyquery signal, NLSA assigns a bucket to it and only computes attention within the bucket. The resulting sparse attention prevents the model from attending to locations thatare noisy and less-informative, while reducing the computa-tional cost from quadratic to asymptotic linear with respectto the spatial size. Extensive experiments validate the effectiveness and efficiency of NLSA. With a few non-local sparseattention modules, our architecture, called non-local sparsenetwork (NLSN), reaches state-of-the-art performance forSISR quantitatively and qualitatively.

![Non-Local Sparse Attention](/Figs/Attention.png)

Non-Local Sparse Attention.

![NLSN](/Figs/NLSN.png)

Non-Local Sparse Network

## Train
### Prepare training data 

1. Download DIV2K training data (800 training + 100 validtion images) from [DIV2K dataset](https://data.vision.ee.ethz.ch/cvl/DIV2K/) or [SNU_CVLab](https://cv.snu.ac.kr/research/EDSR/DIV2K.tar).

2. Specify '--dir_data' based on the HR and LR images path. 

For more informaiton, please refer to [EDSR(PyTorch)](https://github.com/thstkdgus35/EDSR-PyTorch).

### Begin to train

1. (optional) Download pretrained models for our paper.

    Pre-trained models can be downloaded from [Google Drive](https://drive.google.com/drive/folders/1zz2a1ih3euzuH3HvWDN-uSki3USym9Cq?usp=sharing) 

2. Cd to 'src', run the following script to train models.

    **Example command is in the file 'demo.sh'.**

    ```bash
    # Example X2 SR
    python main.py --dir_data ../../ --n_GPUs 4 --rgb_range 1 --chunk_size 144 --n_hashes 4 --save_models --lr 1e-4 --decay 200-400-600-800 --epochs 1000 --chop --save_results --n_resblocks 32 --n_feats 256 --res_scale 0.1 --batch_size 16 --model NLSN --scale 2 --patch_size 96 --save NLSN_x2 --data_train DIV2K

    ```

## Test
### Quick start
1. Download benchmark datasets from [SNU_CVLab](https://cv.snu.ac.kr/research/EDSR/benchmark.tar)

1. (optional) Download pretrained models for our paper.

    All the models can be downloaded from [Google Drive](https://drive.google.com/drive/folders/1zz2a1ih3euzuH3HvWDN-uSki3USym9Cq?usp=sharing) 

2. Cd to 'src', run the following scripts.

    **Example command is in the file 'demo.sh'.**

    ```bash
    # No self-ensemble: NLSN
    # Example X2 SR
    python main.py --dir_data ../../ --model NLSN  --chunk_size 144 --data_test Set5+Set14+B100+Urban100 --n_hashes 4 --chop --save_results --rgb_range 1 --data_range 801-900 --scale 2 --n_feats 256 --n_resblocks 32 --res_scale 0.1  --pre_train model_x2.pt --test_only
    ```

## Citation
If you find the code helpful in your resarch or work, please cite the following papers.
```
  @InProceedings{Mei_2021_CVPR,
    author    = {Mei, Yiqun and Fan, Yuchen and Zhou, Yuqian},
    title     = {Image Super-Resolution With Non-Local Sparse Attention},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2021},
    pages     = {3517-3526}
}
@InProceedings{Lim_2017_CVPR_Workshops,
  author = {Lim, Bee and Son, Sanghyun and Kim, Heewon and Nah, Seungjun and Lee, Kyoung Mu},
  title = {Enhanced Deep Residual Networks for Single Image Super-Resolution},
  booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR) Workshops},
  month = {July},
  year = {2017}
}

```
## Acknowledgements
This code is built on [EDSR (PyTorch)](https://github.com/thstkdgus35/EDSR-PyTorch) and [reformer-pytorch](https://github.com/lucidrains/reformer-pytorch). We thank the authors for sharing their codes.
