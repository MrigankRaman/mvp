<h1 align="center">Model-tuning Via Prompts Makes NLP Models Adversarially Robust</h1>

This is the PyTorch implementation of the MVP paper. This paper uses the [textattack](https://github.com/QData/TextAttack) library

<p align="center">
 <img src="mvp.png" width="700"/>
</p>


## Setup
This repository requires Python 3.8+ and Pytorch 1.11+ but we recommend using Python 3.10 and installing the following libraries

    conda create -n MVP python=3.10
    pip install torch==1.12.1+cu116 --extra-index-url https://download.pytorch.org/whl/cu116
    pip install torch-scatter -f https://data.pyg.org/whl/torch-1.12.0+cu116.html
    pip install textattack[tensorflow]
    
    
## Training and Testing
In the following you can replace mvp after roberta-base with any one of (projectcls, lpft, lpft_dense, clsprompt, mlp_ft) to run the corresponding model

### Training without Adversarial Augmentation :

```
CUDA_VISIBLE_DEVICES=2 bash scripts/train_1_seed.sh 8 boolq roberta-base mvp 20 1e-5 max mean max mean configs/templates_boolq.yaml configs/verbalizer_boolq.yaml mvp_seed_0 textfooler train -1 1 0.1
```

### Training with Adversarial Augmentation :

```
CUDA_VISIBLE_DEVICES=2,3 bash scripts/train_adv_1_seed.sh 8 boolq roberta-base mvp 20 1e-5 max mean max mean configs/templates_boolq.yaml configs/verbalizer_boolq.yaml mvp_adv textfooler train -1 1 0.1 1 l2 1
```

### Testing:
 

```
CUDA_VISIBLE_DEVICES=2 bash scripts/test_1_seed.sh 8 boolq roberta-base mvp 20 1e-5 max mean max mean configs/templates_boolq.yaml configs/verbalizer_boolq.yaml mvp_seed_0 textfooler train -1 1 0.1
```




    
