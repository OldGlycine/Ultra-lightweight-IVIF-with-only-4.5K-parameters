# Ultra-lightweight IVIF with only 4.5K parameters
Inspired from [SFDFusion](https://github.com/lqz2/SFDFusion).
## Environments
```
python 3.9.21
cuda 12.7
```

## Train

The training process needs wandb API key.
The config file is `./configs/cfg.yaml`

```
python train.py
```

## Test
```
python test.py
```

## Datasets
[MSRS-train&test](https://pan.baidu.com/s/1Rj8rXqjq0H13pCKS51qW-A) code=raum

[M3FD-test](https://pan.baidu.com/s/17Ghyd1K-pdvQSBiFDPWrBw) code=pyxy

* The images in M3FD are divided through namelist.txt in the zip file to obtain M3FD-test