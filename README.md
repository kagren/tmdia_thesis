# Overview

This repository contains the code for the master thesis "Self-Supervised Representation Learning for Early Breast Cancer Detection in 
Mammographic Imaging" available [here](https://urn.kb.se/resolve?urn=urn:nbn:se:ltu:diva-106263).

# Installation

```
conda create --name <env> --file requirements.txt
````

The training code assumes Wandb is setup and logged in in the shell environment where you are running the script. If you do not want to use Wandb, comment out the Wandb call from the code.

# Reproduce results

1. Downlaod RSNA and CMMD data sets
2. Preprocess the DICOM images to the same size by cropping to the ROI by using the ```preprocess_rsna.ipynb```and ```preprocess_cmmd.ipynb``` notebooks. Update the paths in the first cell of the notebooks.
3. Train SimSiam and SimClr on the RSNA dataset with and without mixup using ```main_parallel.py```
4. Evaluate embeddings with t-SNE using ```evaluate_tsne.ipynb```
5. Transfer weights and fine tune on the CMMD data set using ```evaluate_finetune_transfer.ipynb```

# Training

The code in ```main_parallel.py``` supports multi GPU training and will automatically use all available GPUs, and supports a number of parameters:

|Parameter   |Values |Description|
|------------|-------|-----------|
|train_path  |string |Path of the train.csv file from the RSNA dataset|
|images_path |string |Path of the preprocessed images from the RSNA dataset|
|batch_size  |int    |Batch size used during training per GPU|
|simclr      |store_true|Set this flag to train SimCLR, if not set, SimSiam will be trained|
|mixup       |store_true|Set this flag to train using the Mixup variant of SimClr or SimSiam|
|model_type |resnet18 or resnet50|The backbone model|
|epochs|int|Amount of epochs to train for|


# Examples

## SimClr with mixup:

```
python main_parallel.py -t data/rsna_bcd/data/train.csv -i /data/rsna_bcd_preprocessed_768_384/ -b 160 --model_type resnet18 --epochs 91 --mixup --simclr
```

## SimSiam with mixup:

```
python main_parallel.py -t data/rsna_bcd/data/train.csv -i /data/rsna_bcd_preprocessed_768_384/ -b 32 --model_type resnet50 --epochs 91 --mixup
```
