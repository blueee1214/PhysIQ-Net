# Deep Underwater Image Quality Assessment via Progressive Physics-aware Multi-Prior Collaboration

This repository contains the official implementation of the following paper:

> **Deep Underwater Image Quality Assessment via Progressive Physics-aware Multi-Prior Collaboration**
>
> Zihan Zhou,  Jiaxue Lan,  Yun Liang,  Weiwei Cai* , Jing Li and Yong Xu and Patrick Le Callet
>
> IEEE Transactions on Circuits and Systems for Video Technology, 2025

## File Structures of the Dataset

- Simply place the images in the dataset in the corresponding folder, the labels are already in "mos.xlsx". The folder structure is as follows. 

```
├───Data/
│   ├───SAUD2.0/
│   │   ├───mos_result/
│   │   │   ├───mos.xlsx
│   │   │   ├───record.txt
│   │   │   └───results.xlsx
│   │   ├───train/
│   │   │   ├───train_dataset.pth
│   │   │   └───...
│   │   ├───test/
│   │   │   ├───test_dataset.pth
│   │   │   └───...
│   │   ├───001_BL-TM.png
│   │   ├───001_GL-net.png
│   │   └───...
│   ├───UID2021/
│   │   └───...
│   └───UWIQA/
│       └───...
│   └───SOTA/
│       └───...
│   ...
```

## Pretrained SyreaNet

- The pretrained checkpoint of SyreaNet can be find in https://github.com/RockWenJJ/SyreaNet. Please rename and add it into the "pretrained_syreanet" folder as follows.

```
├───pretrained_syreanet/
│   ├───__init__.py
│   ├───syreanet.py
│   └───pretrained_syreanet.pth
│   ...
```

## Pretrained RetinexNet

- The pretrained checkpoint of RetinexNet can be find in [https://github.com/aasharma90/RetinexNet_PyTorch.](https://github.com/aasharma90/RetinexNet_PyTorch) Please rename and add it into the "pretrained_Retinex" folder as follows.

```
├───pretrained_Retinex/
│   ├───decomnet.py
│   └───pretrained_Retinex.tar
│   ...
```

## Execution

- Please run "main.py".
- For training, please set "train = True", and set your "data_path".  The file structures of the SAUD2.0, UID2021, UWIQA and SOTA have been given. You can also use your own dataset.
- For testing, please set "train = False", and set your "data_path" and "pretrained_model_path".

## Prepare pretrained models

- You are supposed to download our pretrained model first in the links below and put them in dir ./checkpoints/:[Baidu Disk(pwd: s31p)]( https://pan.baidu.com/s/1lnDe01SBmdA_ZAiyZLRFbw)

## Record and Result

- The record of the training process and the testing results can be found in "**record.txt**", and "**results.xlsx**".

## Citation

If you find the code helpful in your research or work, please cite the following paper.

```
@ARTICLE{11272900,
  author={Zhou, Zihan and Lan, Jiaxue and Liang, Yun and Cai, Weiwei and Li, Jing and Xu, Yong and Callet, Patrick Le},
  journal={IEEE Transactions on Circuits and Systems for Video Technology}, 
  title={Deep Underwater Image Quality Assessment via Progressive Physics-aware Multi-Prior Collaboration}, 
  year={2025}
 }
```

