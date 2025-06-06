from torch.utils.data import Dataset
import torch
from configs import *
import logging
from torchvision.transforms import Compose, Resize
from torchvision import transforms
from pathlib import Path
from typing import Literal
from utils.img_read import img_read
import os
from utils.saliency import Saliency


def check_mask(root: Path, img_list, config: ConfigDict, mode='IVIF'):
    mask_cache = True
    if (root / 'mask').exists():
        for img_name in img_list:
            if not (root / 'mask' / img_name).exists():
                mask_cache = False
                break
    else:
        mask_cache = False
    if mask_cache:
        logging.info('find mask cache in folder, skip saliency detection')
    else:
        logging.info('find no mask cache in folder, start saliency detection')
        saliency = Saliency()
        if mode == 'IVIF':
            saliency.inference(src=root / 'ir', dst=root / 'mask', suffix='png')
        elif mode == 'MIF':
            saliency.inference(src=root / 'MRI', dst=root / 'mask', suffix='png')


class M3FD(Dataset):
    def __init__(self, cfg: ConfigDict, mode: Literal['train', 'val', 'test']):
        super().__init__()
        self.cfg = cfg
        self.mode = mode
        self.img_list = os.listdir(os.path.join(cfg.dataset_root, self.mode, 'ir'))
        logging.info(f'load 210 images')
        self.train_transforms = Compose([Resize((cfg.img_size, cfg.img_size))])

        # if self.mode == 'train' and cfg.have_seg_label == False:
        #     self.ir_path = Path(Path(self.cfg.dataset_root) / 'ir')
        #     self.vi_path = Path(Path(self.cfg.dataset_root) / 'vi')
        #     check_mask(Path(cfg.dataset_root), self.img_list, cfg)
        #     self.mask_path = Path(Path(self.cfg.dataset_root) / 'mask')
        # if self.mode == 'train' and cfg.have_seg_label == True:
        #     self.ir_path = Path(Path(self.cfg.dataset_root) / 'ir')
        #     self.vi_path = Path(Path(self.cfg.dataset_root) / 'vi')
        #     self.mask_path = Path(Path(self.cfg.dataset_root) / 'labels')
        if self.mode == 'test':
            self.ir_path = Path(Path(self.cfg.dataset_root) / 'test' / 'ir')
            self.vi_path = Path(Path(self.cfg.dataset_root) / 'test' / 'vi')

    def __len__(self):
        # return len(self.img_list)
        return 210

    def __getitem__(self, index):
        img_name = self.img_list[index]
        ir_img = img_read(os.path.join(self.ir_path, img_name), mode='L')
        vi_img, vi_cbcr = img_read(os.path.join(self.vi_path, img_name), mode='YCbCr')
        if self.mode == 'train':
            mask = img_read(os.path.join(self.mask_path, img_name), mode='L')
        else:
            mask = None

        if self.mode == 'train':
            ir_img = self.train_transforms(ir_img)
            vi_img = self.train_transforms(vi_img)
            mask = self.train_transforms(mask)
        else:
            _, h, w = ir_img.shape
            if h // 2 != 0 or w // 2 != 0:
                ir_img = ir_img[:, : h // 2 * 2, : w // 2 * 2]
                vi_img = vi_img[:, : h // 2 * 2, : w // 2 * 2]
                vi_cbcr = vi_cbcr[:, : h // 2 * 2, : w // 2 * 2]

        return ir_img, vi_img, mask, img_name, vi_cbcr

    def __collate_fn__(self, batch):
        ir_img_batch, vi_img_batch, mask_batch, img_name_batch, vi_cbcr = zip(*batch)
        ir_img_batch = torch.stack(ir_img_batch, dim=0)
        vi_img_batch = torch.stack(vi_img_batch, dim=0)
        vi_cbcr = torch.stack(vi_cbcr, dim=0)
        if self.mode == 'train':
            mask_batch = torch.stack(mask_batch, dim=0)
        return ir_img_batch, vi_img_batch, mask_batch, img_name_batch, vi_cbcr


class MSRS(Dataset):
    def __init__(self, cfg: ConfigDict, mode: Literal['train', 'test']):
        super().__init__()
        self.cfg = cfg
        self.mode = mode
        self.img_list = os.listdir(os.path.join(cfg.dataset_root, self.mode, 'ir'))
        logging.info(f'load {len(self.img_list)} images')
        self.train_transforms = Compose([
            Resize((cfg.img_size, cfg.img_size)),
            #  transforms.RandomApply([
            #     transforms.ColorJitter(brightness=(0.3, 1.7), contrast=0, saturation=0, hue=0)
            #     ], p=0.5),
            ])

        self.ir_path = Path(Path(self.cfg.dataset_root) / self.mode / 'ir')
        self.vi_path = Path(Path(self.cfg.dataset_root) / self.mode / 'vi')
        if self.mode == 'train' and cfg.have_seg_label == False:
            check_mask(Path(cfg.dataset_root) / 'train', self.img_list, cfg)
            self.mask_path = Path(Path(self.cfg.dataset_root) / self.mode / 'mask')
        if self.mode == 'train' and cfg.have_seg_label == True:
            self.mask_path = Path(Path(self.cfg.dataset_root) / self.mode / 'labels')

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, index):
        img_name = self.img_list[index]
        ir_img = img_read(os.path.join(self.ir_path, img_name), mode='L')
        vi_img, vi_cbcr = img_read(os.path.join(self.vi_path, img_name), mode='YCbCr')
        if self.mode == 'train':
            mask = img_read(os.path.join(self.mask_path, img_name), mode='L')
        else:
            mask = None

        if self.mode == 'train':
            ir_img = self.train_transforms(ir_img)
            vi_img = self.train_transforms(vi_img)
            mask = self.train_transforms(mask)
            # vi_cbcr = self.train_transforms(vi_cbcr)
        else:
            _, h, w = ir_img.shape
            if h // 2 != 0 or w // 2 != 0:
                ir_img = ir_img[:, : h // 2 * 2, : w // 2 * 2]
                vi_img = vi_img[:, : h // 2 * 2, : w // 2 * 2]
                vi_cbcr = vi_cbcr[:, : h // 2 * 2, : w // 2 * 2]

        return ir_img, vi_img, mask, img_name, vi_cbcr

    def __collate_fn__(self, batch):
        ir_img_batch, vi_img_batch, mask_batch, img_name_batch, vi_cbcr = zip(*batch)
        ir_img_batch = torch.stack(ir_img_batch, dim=0)
        vi_img_batch = torch.stack(vi_img_batch, dim=0)
        vi_cbcr = torch.stack(vi_cbcr, dim=0)
        if self.mode == 'train':
            mask_batch = torch.stack(mask_batch, dim=0)
        return ir_img_batch, vi_img_batch, mask_batch, img_name_batch, vi_cbcr

class TNO(Dataset):
    def __init__(self, cfg: ConfigDict, mode: Literal['train', 'test']):
        super().__init__()
        self.cfg = cfg
        self.mode = mode
        self.img_list = os.listdir(os.path.join(cfg.dataset_root, self.mode, 'ir'))
        self.vi_list = os.listdir(os.path.join(cfg.dataset_root, self.mode, 'vi'))
        logging.info(f'load {len(self.img_list)} images')
        self.train_transforms = Compose([Resize((cfg.img_size, cfg.img_size))])

        self.ir_path = Path(Path(self.cfg.dataset_root) / self.mode / 'ir')
        self.vi_path = Path(Path(self.cfg.dataset_root) / self.mode / 'vi')
        if self.mode == 'train' and cfg.have_seg_label == False:
            check_mask(Path(cfg.dataset_root) / 'train', self.img_list, cfg)
            self.mask_path = Path(Path(self.cfg.dataset_root) / self.mode / 'mask')
        if self.mode == 'train' and cfg.have_seg_label == True:
            self.mask_path = Path(Path(self.cfg.dataset_root) / self.mode / 'labels')

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, index):
        img_name = self.img_list[index]
        vi_name = self.vi_list[index]
        ir_img = img_read(os.path.join(self.ir_path, img_name), mode='L')
        vi_img, vi_cbcr = img_read(os.path.join(self.vi_path, vi_name), mode='YCbCr')
        if self.mode == 'train':
            mask = img_read(os.path.join(self.mask_path, img_name), mode='L')
        else:
            mask = None

        if self.mode == 'train':
            ir_img = self.train_transforms(ir_img)
            vi_img = self.train_transforms(vi_img)
            mask = self.train_transforms(mask)
            vi_cbcr = self.train_transforms(vi_cbcr)
        else:
            _, h, w = ir_img.shape
            if h // 2 != 0 or w // 2 != 0:
                ir_img = ir_img[:, : h // 2 * 2, : w // 2 * 2]
                vi_img = vi_img[:, : h // 2 * 2, : w // 2 * 2]
                vi_cbcr = vi_cbcr[:, : h // 2 * 2, : w // 2 * 2]

        return ir_img, vi_img, mask, img_name, vi_cbcr

    def __collate_fn__(self, batch):
        ir_img_batch, vi_img_batch, mask_batch, img_name_batch, _ = zip(*batch)
        ir_img_batch = torch.stack(ir_img_batch, dim=0)
        vi_img_batch = torch.stack(vi_img_batch, dim=0)
        if self.mode == 'train':
            mask_batch = torch.stack(mask_batch, dim=0)
        return ir_img_batch, vi_img_batch, mask_batch, img_name_batch

class RoadScene(Dataset):
    def __init__(self, cfg: ConfigDict, mode: Literal['train', 'val', 'test']):
        super().__init__()
        self.cfg = cfg
        self.mode = mode
        self.img_list = Path(Path(self.cfg.dataset_root) / f'{mode}.txt').read_text().splitlines()
        logging.info(f'load {len(self.img_list)} images')
        self.train_transforms = Compose([Resize((cfg.img_size, cfg.img_size))])

        if self.mode == 'train' and cfg.have_seg_label == False:
            self.ir_path = Path(Path(self.cfg.dataset_root) / 'ir')
            self.vi_path = Path(Path(self.cfg.dataset_root) / 'vi')
            check_mask(Path(cfg.dataset_root), self.img_list, cfg)
            self.mask_path = Path(Path(self.cfg.dataset_root) / 'mask')
        if self.mode == 'train' and cfg.have_seg_label == True:
            self.ir_path = Path(Path(self.cfg.dataset_root) / 'ir')
            self.vi_path = Path(Path(self.cfg.dataset_root) / 'vi')
            self.mask_path = Path(Path(self.cfg.dataset_root) / 'labels')
        if self.mode == 'test':
            self.ir_path = Path(Path(self.cfg.dataset_root) / 'test' / 'ir')
            self.vi_path = Path(Path(self.cfg.dataset_root) / 'test' / 'vi')

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, index):
        img_name = self.img_list[index]
        ir_img = img_read(os.path.join(self.ir_path, img_name), mode='L')
        vi_img, vi_cbcr = img_read(os.path.join(self.vi_path, img_name), mode='YCbCr')
        if self.mode == 'train':
            mask = img_read(os.path.join(self.mask_path, img_name), mode='L')
        else:
            mask = None

        if self.mode == 'train':
            ir_img = self.train_transforms(ir_img)
            vi_img = self.train_transforms(vi_img)
            mask = self.train_transforms(mask)
        else:
            _, h, w = ir_img.shape
            if h // 2 != 0 or w // 2 != 0:
                ir_img = ir_img[:, : h // 2 * 2, : w // 2 * 2]
                vi_img = vi_img[:, : h // 2 * 2, : w // 2 * 2]
                vi_cbcr = vi_cbcr[:, : h // 2 * 2, : w // 2 * 2]

        return ir_img, vi_img, mask, img_name, vi_cbcr

    def __collate_fn__(self, batch):
        ir_img_batch, vi_img_batch, mask_batch, img_name_batch, vi_cbcr = zip(*batch)
        ir_img_batch = torch.stack(ir_img_batch, dim=0)
        vi_img_batch = torch.stack(vi_img_batch, dim=0)
        vi_cbcr = torch.stack(vi_cbcr, dim=0)
        if self.mode == 'train':
            mask_batch = torch.stack(mask_batch, dim=0)
        return ir_img_batch, vi_img_batch, mask_batch, img_name_batch, vi_cbcr

class PET_MRI(Dataset):
    def __init__(self, cfg: ConfigDict, mode: Literal['train', 'test']):
        super().__init__()
        self.cfg = cfg
        self.mode = mode
        self.img_list = os.listdir(os.path.join(cfg.dataset_root, self.mode, 'MRI'))
        logging.info(f'load {len(self.img_list)} images')
        self.train_transforms = Compose([
            Resize((cfg.img_size, cfg.img_size)),
            ])

        self.mri_path = Path(Path(self.cfg.dataset_root) / self.mode / 'MRI')
        self.pet_path = Path(Path(self.cfg.dataset_root) / self.mode / 'PET')
        if self.mode == 'train':
            check_mask(Path(cfg.dataset_root) / 'train', self.img_list, cfg, mode='MIF')
            self.mask_path = Path(Path(self.cfg.dataset_root) / self.mode / 'mask')

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, index):
        img_name = self.img_list[index]
        mri_img = img_read(os.path.join(self.mri_path, img_name), mode='L')
        pet_img, pet_cbcr = img_read(os.path.join(self.pet_path, img_name), mode='YCbCr')
        if self.mode == 'train':
            mask = img_read(os.path.join(self.mask_path, img_name), mode='L')
        else:
            mask = None

        if self.mode == 'train':
            mri_img = self.train_transforms(mri_img)
            pet_img = self.train_transforms(pet_img)
            mask = self.train_transforms(mask)
            pet_cbcr = self.train_transforms(pet_cbcr)
        else:
            _, h, w = mri_img.shape
            if h // 2 != 0 or w // 2 != 0:
                mri_img = mri_img[:, : h // 2 * 2, : w // 2 * 2]
                pet_img = pet_img[:, : h // 2 * 2, : w // 2 * 2]
                pet_cbcr = pet_cbcr[:, : h // 2 * 2, : w // 2 * 2]

        return mri_img, pet_img, mask, img_name, pet_cbcr

    def __collate_fn__(self, batch):
        mri_img_batch, pet_img_batch, mask_batch, img_name_batch, pet_cbcr = zip(*batch)
        mri_img_batch = torch.stack(mri_img_batch, dim=0)
        pet_img_batch = torch.stack(pet_img_batch, dim=0)
        pet_cbcr = torch.stack(pet_cbcr, dim=0)
        if self.mode == 'train':
            mask_batch = torch.stack(mask_batch, dim=0)
        return mri_img_batch, pet_img_batch, mask_batch, img_name_batch, pet_cbcr

class SPECT_MRI(Dataset):
    def __init__(self, cfg: ConfigDict, mode: Literal['train', 'test']):
        super().__init__()
        self.cfg = cfg
        self.mode = mode
        self.img_list = os.listdir(os.path.join(cfg.dataset_root, self.mode, 'MRI'))
        logging.info(f'load {len(self.img_list)} images')
        self.train_transforms = Compose([
            Resize((cfg.img_size, cfg.img_size)),
            ])

        self.mri_path = Path(Path(self.cfg.dataset_root) / self.mode / 'MRI')
        self.spect_path = Path(Path(self.cfg.dataset_root) / self.mode / 'SPECT')
        if self.mode == 'train':
            check_mask(Path(cfg.dataset_root) / 'train', self.img_list, cfg)
            self.mask_path = Path(Path(self.cfg.dataset_root) / self.mode / 'mask')

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, index):
        img_name = self.img_list[index]
        mri_img = img_read(os.path.join(self.mri_path, img_name), mode='L')
        spect_img, spect_cbcr = img_read(os.path.join(self.spect_path, img_name), mode='YCbCr')
        if self.mode == 'train':
            mask = img_read(os.path.join(self.mask_path, img_name), mode='L')
        else:
            mask = None

        if self.mode == 'train':
            mri_img = self.train_transforms(mri_img)
            spect_img = self.train_transforms(spect_img)
            mask = self.train_transforms(mask)
            spect_cbcr = self.train_transforms(spect_cbcr)
        else:
            _, h, w = mri_img.shape
            if h // 2 != 0 or w // 2 != 0:
                mri_img = mri_img[:, : h // 2 * 2, : w // 2 * 2]
                spect_img = spect_img[:, : h // 2 * 2, : w // 2 * 2]
                spect_cbcr = spect_cbcr[:, : h // 2 * 2, : w // 2 * 2]

        return mri_img, spect_img, mask, img_name, spect_cbcr

    def __collate_fn__(self, batch):
        mri_img_batch, spect_img_batch, mask_batch, img_name_batch, spect_cbcr = zip(*batch)
        mri_img_batch = torch.stack(mri_img_batch, dim=0)
        spect_img_batch = torch.stack(spect_img_batch, dim=0)
        spect_cbcr = torch.stack(spect_cbcr, dim=0)
        if self.mode == 'train':
            mask_batch = torch.stack(mask_batch, dim=0)
        return mri_img_batch, spect_img_batch, mask_batch, img_name_batch, spect_cbcr

class CT_MRI(Dataset):
    def __init__(self, cfg: ConfigDict, mode: Literal['train', 'test']):
        super().__init__()
        self.cfg = cfg
        self.mode = mode
        self.img_list = os.listdir(os.path.join(cfg.dataset_root, self.mode, 'MRI'))
        logging.info(f'load {len(self.img_list)} images')
        self.train_transforms = Compose([
            Resize((cfg.img_size, cfg.img_size)),
            ])

        self.mri_path = Path(Path(self.cfg.dataset_root) / self.mode / 'MRI')
        self.ct_path = Path(Path(self.cfg.dataset_root) / self.mode / 'CT')
        if self.mode == 'train':
            check_mask(Path(cfg.dataset_root) / 'train', self.img_list, cfg)
            self.mask_path = Path(Path(self.cfg.dataset_root) / self.mode / 'mask')

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, index):
        img_name = self.img_list[index]
        mri_img = img_read(os.path.join(self.mri_path, img_name), mode='L')
        ct_img, ct_cbcr = img_read(os.path.join(self.ct_path, img_name), mode='YCbCr')
        if self.mode == 'train':
            mask = img_read(os.path.join(self.mask_path, img_name), mode='L')
        else:
            mask = None

        if self.mode == 'train':
            mri_img = self.train_transforms(mri_img)
            ct_img = self.train_transforms(ct_img)
            mask = self.train_transforms(mask)
            ct_cbcr = self.train_transforms(ct_cbcr)
        else:
            _, h, w = mri_img.shape
            if h // 2 != 0 or w // 2 != 0:
                mri_img = mri_img[:, : h // 2 * 2, : w // 2 * 2]
                ct_img = ct_img[:, : h // 2 * 2, : w // 2 * 2]
                ct_cbcr = ct_cbcr[:, : h // 2 * 2, : w // 2 * 2]

        return mri_img, ct_img, mask, img_name, ct_cbcr

    def __collate_fn__(self, batch):
        mri_img_batch, ct_img_batch, mask_batch, img_name_batch, ct_cbcr = zip(*batch)
        mri_img_batch = torch.stack(mri_img_batch, dim=0)
        ct_img_batch = torch.stack(ct_img_batch, dim=0)
        ct_cbcr = torch.stack(ct_cbcr, dim=0)
        if self.mode == 'train':
            mask_batch = torch.stack(mask_batch, dim=0)
        return mri_img_batch, ct_img_batch, mask_batch, img_name_batch, ct_cbcr


if __name__ == '__main__':
    import yaml

    config = yaml.safe_load(open('./configs/cfg.yaml'))
    cfg = from_dict(config)
    train_dataset = MSRS(cfg, 'train')
    # 绘制数据集
    import matplotlib.pyplot as plt

    for i in range(3):
        ir, vi, mask, img_name, cbcr = train_dataset[i]
        ir = ir.squeeze().numpy()
        vi = vi.squeeze().numpy()
        mask = mask.squeeze().numpy()
        plt.subplot(231)
        plt.imshow(ir, cmap='gray')
        plt.subplot(232)
        plt.imshow(vi, cmap='gray')
        plt.subplot(233)
        plt.imshow(mask, cmap='gray')
        plt.subplot(234)
        plt.imshow(cbcr[0,:,:], cmap='gray')
        plt.subplot(235)
        plt.imshow(cbcr[1,:,:], cmap='gray')
        plt.savefig(f'./{img_name}.png')
