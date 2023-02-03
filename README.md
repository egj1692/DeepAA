# DeepAA

I try to implement Deep AutoAugment in windows

### Install required packages

1. Create anaconda virtual eviroment
'''shell
conda create -n deepaa python=3.7.7
conda activate deepaa
'''

2. Install Tensorflow and PyTorch
'''shell
pip install tensorflow-gpu==2.5
conda install cudnn=8.1 cudatoolkit=11.2 -c conda-forge
pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html
'''

3. Install other dependencies.
```shell
pip install -r requirements.txt
```

### Run augmentation policy search on CIFAR-10/100. 

```shell
CUDA_VISIBLE_DEVICES=0,1 python DeepAA_search.py --dataset cifar10 --n_classes 10 --use_model WRN_40_2 --n_policies 6 --search_bno 1024 --pretrain_lr 0.1 --seed 1 --batch_size 128 --test_batch_size 512 --policy_lr 0.025 --l_mags 13 --use_pool --pretrain_size 5000 --nb_epochs 45 --EXP_G 16 --EXP_gT_factor=4 --train_same_labels 16
```
