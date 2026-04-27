# SAPL
## Requirements
The code implementation of **SAPL** mainly based on [PyTorch](https://pytorch.org/). All of our experiments run and test in Python 3.11.13. First install the dependencies.

Either manually:
```
conda install pytorch torchvision -c pytorch
conda install matplotlib torchmetrics -c conda-forge
```
Additionally, we use Weights & Biases (W&B) to keep track and organize the results of experiments. You may need to follow the online documentation of W&B to quickstart. To run these codes, sign up an online account to track experiments or create a local wandb server using docker (recommended).

## Dataset
Please follow the instructions SCTransNet(https://github.com/xdFai/SCTransNet) to construct the datasets. For the dataset split, we follow text-IRSTD, please download the split files from here(https://github.com/YangBo0411/infrared-small-target/tree/main/data)

## Preparing
Please extract clip features before training:

```
$python extract_clip_features.py
```

## Training Script
```
$ python train.py
```
