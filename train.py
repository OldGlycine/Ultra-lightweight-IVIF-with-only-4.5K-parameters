import yaml
import lightning as L
import random
from configs import *
import dataset
import numpy as np
import torch
from torch.utils.data import DataLoader
import argparse
import wandb
from lightning.pytorch.profilers import AdvancedProfiler
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import RichProgressBar
from lightning.pytorch.callbacks.progress.rich_progress import RichProgressBarTheme
from lightning.pytorch.tuner import Tuner
import importlib

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def get_model(arch_name):
    module = importlib.import_module(f"archs.{arch_name}")
    return getattr(module, arch_name)

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    torch.Generator().manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark =False

def train(cfg_path, auth, mode='train'):
    config = yaml.safe_load(open(cfg_path))
    cfg = from_dict(config)
    set_seed(cfg.seed)

    # wandb
    if mode == 'train':
        wandb.login(key=auth)
    wandb_logger = WandbLogger(project=cfg.project_name, name=cfg.dataset_name + '_' + cfg.exp_name, save_dir='./wandb', config=cfg, mode=cfg.wandb_mode)

    # Get datasets
    train_d = getattr(dataset, cfg.dataset_name)
    train_dataset = train_d(cfg, 'train')

    trainloader = DataLoader(
        train_dataset, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers, collate_fn=train_dataset.__collate_fn__, pin_memory=True
    )

    # profiler
    # profiler = AdvancedProfiler(dirpath="./model/", filename="perf_logs_" + cfg.exp_name)
    # bar
    progress_bar = RichProgressBar(
    theme=RichProgressBarTheme(
        description="green_yellow",
        progress_bar="green1",
        progress_bar_finished="green1",
        progress_bar_pulse="#6206E0",
        batch_progress="green_yellow",
        time="grey82",
        processing_speed="grey82",
        metrics="grey82",
        metrics_text_delimiter="\n",
        metrics_format=".3e",
    )
)

    # Model
    net = get_model((cfg.exp_name).split('_')[0])
    net = net(cfg, args)
    
    trainer = L.Trainer(default_root_dir='./model/' + cfg.exp_name,
                        # profiler=profiler,
                        logger=wandb_logger,
                        # precision='bf16-mixed',
                        callbacks=progress_bar, 
                        max_epochs=cfg.num_epochs,
                        )
    if mode == 'train':
        trainer.fit(model=net, 
                    train_dataloaders=trainloader)
        # save
        trainer.save_checkpoint('./model/' + cfg.exp_name + '.pth')
        torch.cuda.empty_cache()
    elif mode == 'tune':
        # Create a Tuner
        tuner = Tuner(trainer)

        # finds learning rate automatically
        # sets hparams.lr or hparams.learning_rate to that learning rate
        lr_finder = tuner.lr_find(net, train_dataloaders=trainloader, num_training=50)
    
        # Results can be found in
        print(lr_finder.results)
        # Plot with
        fig = lr_finder.plot(suggest=True)
        fig.show()
        new_lr = lr_finder.suggestion()
        print(new_lr)

        # batch
        # batch_finder = tuner.scale_batch_size(net, train_dataloaders=trainloader, mode="binsearch")
        # print(batch_finder)
        
        input('Press any key to exit')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', default='configs/cfg.yaml', help='config file path')
    parser.add_argument('--auth', default='Your wandb api key', help='wandb auth api key')
    args = parser.parse_args()
    train(args.cfg, args.auth)
    # train(args.cfg, args.auth, mode='tune')
    print('done')

