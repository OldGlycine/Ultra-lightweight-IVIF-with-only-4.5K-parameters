import torch
import torch.nn as nn
import numpy as np
import lightning as L
from utils.loss import *
import kornia
from kornia.metrics import AverageMeter
from utils.img_read import *
import time

class Att_Block_lite(nn.Module):
    def __init__(self, in_channels, out_channels, group_ratio=1):
        super().__init__()
        assert in_channels % group_ratio == 0
        self.att = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1), nn.Hardsigmoid(True))
        # self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        att = self.att(x)
        x = x * att
        return x

class feature_extra(nn.Module):
    def __init__(self, channels, kernel_size=3, padding=1, stride=1, dilation=1, groups=1):
        super(feature_extra, self).__init__()
        self.conv1 = nn.Conv2d(
            channels, channels, kernel_size=kernel_size, padding=padding, stride=stride, dilation=dilation, groups=channels, bias=False
        )
        self.conv2 = nn.Conv2d(
            channels, channels, kernel_size=kernel_size, padding=padding, stride=stride, dilation=dilation, groups=channels, bias=False
        )

    def forward(self, x):
        sx = self.conv1(x)
        sy = self.conv2(x)
        x = torch.abs(sx) + torch.abs(sy)
        return x

class EO(nn.Module):
    def __init__(self, in_channels, hidden, out_channels, kernel_size=3, stride=1, padding=1, dilation=1, groups=1):
        super(EO, self).__init__()
        self.ir_enc = nn.Sequential(nn.Conv2d(in_channels, hidden, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups), nn.ReLU(True))
        self.vi_enc = nn.Sequential(nn.Conv2d(in_channels, hidden, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups), nn.ReLU(True))
        self.ir_dec = nn.Sequential(nn.Conv2d(hidden, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups), nn.ReLU(True))
        self.vi_dec = nn.Sequential(nn.Conv2d(hidden, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups), nn.ReLU(True))
        self.ir_fea = feature_extra(hidden)
        self.vi_fea = feature_extra(hidden)
        self.init_weights()
        self.ir_att1 = Att_Block_lite(hidden, hidden)
        self.ir_att2 = Att_Block_lite(hidden, hidden)
        self.vi_att1 = Att_Block_lite(hidden, hidden)
        self.vi_att2 = Att_Block_lite(hidden, hidden)
        
        
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, ir ,vi):
        ir_embed, vi_embed = self.ir_enc(ir), self.vi_enc(vi)
        chimera = ir_embed + vi_embed
        ir_coarse, vi_coarse = self.ir_att1(ir_embed), self.vi_att1(vi_embed)
        ir_fine, vi_fine = self.ir_att2(chimera), self.vi_att2(chimera)
        ir_feature, vi_feature = self.ir_fea(ir_embed), self.vi_fea(vi_embed)
        return self.ir_dec(ir_coarse + ir_feature + ir_fine), self.vi_dec(vi_coarse + vi_feature + vi_fine), ir_embed, vi_embed

class Fuse_out(nn.Module):
    def __init__(self, in_channels=3, out_channels=1): # dim=24
        super(Fuse_out, self).__init__()
        self.out_conv1 = nn.Sequential(nn.Conv2d(18, 16, kernel_size=1, stride=1), nn.ReLU(True),)
        self.out_conv2 = nn.Sequential(nn.Conv2d(16, 8, kernel_size=3, stride=1, padding=1), nn.ReLU(True),)
        self.out_conv3 = nn.Sequential(nn.Conv2d(8, out_channels, kernel_size=3, stride=1, padding=1),nn.Hardtanh())
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, ir, vi, ir_embed, vi_embed):
        x1 = torch.cat([ir, ir_embed, vi, vi_embed], dim=1)
        x2 = self.out_conv1(x1)
        x3 = self.out_conv2(x2)
        return self.out_conv3(x3)

class Fuse87(L.LightningModule):
    def __init__(self, cfg=None, args=None):
        super().__init__()
        # utils
        self.cfg = cfg
        self.args = args
        self.logs = {}
        self.total_loss_meter, self.content_loss_meter, self.ssim_loss_meter, self.saliency_loss_meter, self.recons_loss_meter, self.fre_loss_meter = AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter()
        self.learning_rate = cfg.lr_i
        self.batch_size = cfg.batch_size
        self.time_list = []

        self.channel = 8
        self.layer = EO(in_channels=1, hidden=8, out_channels=1)
        self.fus_out = Fuse_out()

        # self.save_hyperparameters()

    def forward(self, ir, vi):
        ir, vi, ir_embed, vi_embed = self.layer(ir, vi)
        fus = self.fus_out(ir, vi, ir_embed, vi_embed)
        # fus = (fus - torch.min(fus)) / (torch.max(fus) - torch.min(fus))
        return fus, ir, vi
    
    def training_step(self, batch, batch_indx):
        total_loss, content_loss, ssim_loss, saliency_loss, recons_loss = self._get_losses(batch)
        self.total_loss_meter.update(total_loss.item())
        self.content_loss_meter.update(content_loss.item())
        self.ssim_loss_meter.update(ssim_loss.item())
        self.saliency_loss_meter.update(saliency_loss.item())
        self.recons_loss_meter.update(recons_loss.item())
        optimizer = self.optimizers()
        self.logs |= {
            'total_loss': self.total_loss_meter.avg,
            'content_loss': self.content_loss_meter.avg,
            'ssim_loss': self.ssim_loss_meter.avg,
            'saliency_loss': self.saliency_loss_meter.avg,
            'recons_loss': self.recons_loss_meter.avg,
            'lr': optimizer.param_groups[0]["lr"],
        }

        self.log_dict(self.logs)

        return total_loss
    
    def test_step(self, batch, batch_idx):
        ir, vi, _, img_name, vi_cbcr = batch
        ts = time.time()
        fus_data, _, _ = self(ir, vi)
        fus_data = (fus_data - torch.min(fus_data)) / (torch.max(fus_data) - torch.min(fus_data))
        te = time.time()
        self.time_list.append(te - ts)
        if self.args.mode == 'gray':
            fi = np.squeeze((fus_data * 255).detach().to(torch.float).cpu().numpy()).astype(np.uint8)
            img_save(fi, img_name[0], self.args.out_dir)
        elif self.args.mode == 'RGB':
            vi_cbcr = vi_cbcr.to(self.device)
            fi = torch.cat((fus_data, vi_cbcr), dim=1)
            fi = ycbcr_to_rgb(fi)
            fi = tensor_to_image(fi) * 255
            fi = fi.astype(np.uint8)
            img_save(fi, img_name[0], self.args.out_dir + 'RGB', mode='RGB')
        return fi
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        lr_func = lambda x: (1 - x / self.cfg.num_epochs) * (1 - self.cfg.lr_f) + self.cfg.lr_f
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_func)
        return {
            'optimizer' : optimizer,
            'lr_scheduler' : scheduler
        }
    
    def get_avg_times(self):
        avg_time = np.round(np.mean(self.time_list[1:]), 6)
        print(f'fusing images done!')
        print(f'time: {avg_time}s')
        return avg_time

    def _get_losses(self, batch):
        # data
        ir, vi, mask, _, _ = batch

        # results
        fus, ir_d, vi_d = self(ir, vi)

        # loss
        loss_ssim = kornia.losses.SSIMLoss(window_size=11)
        loss_grad_pixel = PixelGradLoss()
        # conten_loss
        content_loss = loss_grad_pixel(vi, ir, fus)
        # SSIM-loss
        ssim_loss_v = loss_ssim(vi, fus)
        ssim_loss_i = loss_ssim(ir, fus)
        ssim_loss = ssim_loss_i + ssim_loss_v
        # saliency_loss
        saliency_loss = cal_saliency_loss(fus, ir, vi, mask)
        # recons loss
        mse_loss_VF = 5 * loss_ssim(vi, vi_d) + nn.MSELoss()(vi, vi_d)
        mse_loss_IF = 5 * loss_ssim(ir, ir_d) + nn.MSELoss()(ir, ir_d)
        recons_loss = 2 * mse_loss_VF + 2 * mse_loss_IF

        total_loss = self.cfg.coeff_content * content_loss + self.cfg.coeff_ssim * ssim_loss + self.cfg.coeff_saliency * saliency_loss + self.cfg.coeff_recons * recons_loss
        return total_loss, content_loss, ssim_loss, saliency_loss, recons_loss
