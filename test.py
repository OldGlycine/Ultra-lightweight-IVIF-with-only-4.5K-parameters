import warnings
import yaml
from configs import *
import dataset
from torch.utils.data import DataLoader
import argparse
import os
from lightning.pytorch.callbacks import RichProgressBar
from lightning.pytorch.callbacks.progress.rich_progress import RichProgressBarTheme
from tqdm import tqdm
from utils.img_read import img_read
from utils.evaluator import Evaluator
import cv2
import pandas as pd
import importlib
import numpy as np
import torch
import lightning as L
from kornia.metrics import AverageMeter

warnings.filterwarnings("ignore")
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = 'cuda' if torch.cuda.is_available() else 'cpu'

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

def get_model(arch_name):
    module = importlib.import_module(f"archs.{arch_name}")
    return getattr(module, arch_name)

def test(cfg, args, dataset_name):
    # model
    net = get_model((cfg.exp_name).split('_')[0])
    net = net.load_from_checkpoint(checkpoint_path=args.ckpt_path, map_location=device, cfg=cfg, args=args)
    # wb = torch.load('./model/Fuse12-.pth')
    # net.load_state_dict(wb)
    net.eval()
    
    # dataset
    test_d = getattr(dataset, dataset_name)
    test_dataset = test_d(cfg, 'test')

    testloader = DataLoader(
        test_dataset, batch_size=1, shuffle=False, num_workers=cfg.num_workers, collate_fn=test_dataset.__collate_fn__, pin_memory=True
    )
    fuse_out_folder = args.out_dir
    if not os.path.exists(fuse_out_folder):
        os.makedirs(fuse_out_folder)

    trainer = L.Trainer(
                        # precision='bf16-mixed',
                        callbacks=progress_bar,
                        # logger=WandbLogger(dir=args.out_dir)
                        )
    trainer.test(model=net, dataloaders=testloader)
    avg_time = net.get_avg_times()
    torch.cuda.empty_cache()
    return avg_time

def eval(fuse_out_folder, avg_time):
    test_d = getattr(dataset, cfg.dataset_name)
    test_dataset = test_d(cfg, 'test')
    testloader = DataLoader(
        test_dataset, batch_size=1, shuffle=False, num_workers=cfg.num_workers, collate_fn=test_dataset.__collate_fn__, pin_memory=True
    )
    metric_result = [AverageMeter() for _ in range(6)]

    print(f'evaluating images ...')
    iter = tqdm(testloader, total=len(testloader), ncols=80)

    for data_ir, data_vi, _, img_name, _ in iter:
        ir = data_ir.squeeze() * 255
        vi = data_vi.squeeze() * 255
        fi = img_read(os.path.join(fuse_out_folder, img_name[0]), 'L').squeeze() * 255
        h, w = fi.shape
        if h // 2 != 0 or w // 2 != 0:
            fi = fi[: h // 2 * 2, : w // 2 * 2]
        if fi.shape != ir.shape or fi.shape != vi.shape:
            fi = cv2.resize(fi, (ir.shape[1], ir.shape[0]))
        # print(ir.shape, vi.shape, fi.shape)
        # print(type(fi), type(vi),type(ir))
        metric_result[0].update(Evaluator.EN(fi))
        metric_result[1].update(Evaluator.SD(fi))
        metric_result[2].update(Evaluator.SF(fi))
        metric_result[3].update(Evaluator.MI(fi, ir, vi))
        metric_result[4].update(Evaluator.VIFF(fi, ir, vi))
        metric_result[5].update(Evaluator.Qabf(fi, ir, vi))

    # 结果写入文件
    # with open(f'{fuse_out_folder}_result.txt', 'w') as f:
    #     f.write('EN: ' + str(np.round(metric_result[0].avg, decimals=3)) + '\n')
    #     f.write('SD: ' + str(np.round(metric_result[1].avg, decimals=3)) + '\n')
    #     f.write('SF: ' + str(np.round(metric_result[2].avg, decimals=3)) + '\n')
    #     f.write('MI: ' + str(np.round(metric_result[3].avg, decimals=3)) + '\n')
    #     f.write('VIF: ' + str(np.round(metric_result[4].avg, decimals=3)) + '\n')
    #     f.write('Qabf: ' + str(np.round(metric_result[5].avg, decimals=3)) + '\n')
    csv_dir = fuse_out_folder.split('/')
    csv_dir[-1] = 'result.csv'
    csv_dir = '/'.join(csv_dir)
    fuse_result = pd.read_csv(csv_dir, index_col=0)
    df = pd.DataFrame({'EN':[np.round(metric_result[0].avg, decimals=3)],
                       'SD':[np.round(metric_result[1].avg, decimals=3)],
                       'SF':[np.round(metric_result[2].avg, decimals=3)],
                       'MI':[np.round(metric_result[3].avg, decimals=3)],
                       'VIF':[np.round(metric_result[4].avg, decimals=3)],
                        'Qabf':[np.round(metric_result[5].avg, decimals=3)],
                        'time':[np.round(avg_time, decimals=6)]},
                        index=[cfg.exp_name])
    fuse_result = pd.concat([fuse_result, df])
    fuse_result.to_csv(csv_dir, index=True)
    print(f'writing results done!')
    print("\n" * 2 + "=" * 80)
    print("The test result :")
    print("\t\t EN\t SD\t SF\t MI\tVIF\tQabf")
    print(
        'result:\t'
        + '\t'
        + str(np.round(metric_result[0].avg, 3))
        + '\t'
        + str(np.round(metric_result[1].avg, 3))
        + '\t'
        + str(np.round(metric_result[2].avg, 3))
        + '\t'
        + str(np.round(metric_result[3].avg, 3))
        + '\t'
        + str(np.round(metric_result[4].avg, 3))
        + '\t'
        + str(np.round(metric_result[5].avg, 3))
    )
    print("=" * 80)

        
if __name__ == '__main__':
    config = yaml.safe_load(open('configs/cfg.yaml'))
    cfg = from_dict(config)
    diction = ['RoadScene' , 'MSRS', 'M3FD']
    for string in diction:
        parse = argparse.ArgumentParser()
        parse.add_argument('--ckpt_path', type=str, default=f'model/{cfg.exp_name}.pth')
        parse.add_argument('--out_dir', type=str, default=f'test_result/{string}/{cfg.exp_name}')
        parse.add_argument('--mode', type=str, default='gray')
        args = parse.parse_args()
        cfg.dataset_root = './' + string
        cfg.dataset_name = string

        avg_time = test(cfg, args, string)
        eval(args.out_dir, avg_time)