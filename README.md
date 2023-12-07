# SAM for Sperm Cell Segmentation

> This repo reimplement by [Officially SAM Repo](https://github.com/facebookresearch/segment-anything)

## Dataset prepration

- Suggest for single cell segmentation

- Change image path for your own image path

## Pre-trained Models

- Download pretrained models on [GoogleDrive](https://cuhko365-my.sharepoint.com/:f:/g/personal/223010087_link_cuhk_edu_cn/EjIVHv5WBxVBsqJ9xa-Q9KEBZ2jV2N7VphbiDA-R1-2Xiw?e=HrUdPb)

- Including default or vit_h, vit_l and vit_b

## Clone Repo and create virtual env

```bash
git clone git@github.com:SimonHanYANG/SAM4Sperm.git
```

- Create Virtual Env
```bash
conda create -n samsperm python=3.9

conda activate samsperm

pip install -r requirements.txt

# pip install segment-anything
pip install git+https://github.com/facebookresearch/segment-anything.git
```

## Run Demo

```bash
python demo.py
```
