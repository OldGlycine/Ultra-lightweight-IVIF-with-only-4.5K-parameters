from torchsummary import summary
from torchinfo import summary as sm
from archs.Fuse87 import *
from archs.DIDFuse import *
from archs.SFDFusion import * 
import yaml
from configs import *
from modules_backup import * 
from thop import profile
device = 'cuda' if torch.cuda.is_available() else 'cpu'


if __name__ == '__main__':
    config = yaml.safe_load(open('configs/cfg.yaml'))
    cfg = from_dict(config)
    # model = Fuse87(cfg=cfg)
    model = SFDFusion(cfg=cfg)
    input1 = torch.randn(1, 1, 320, 320) 
    input2 = torch.randn(1, 1, 320, 320) 
    print(input1.dtype)
    inputs = (input1, input2)
    # flops, params = profile(model, inputs)
    # print('FLOPs = ' + str(flops / 1000**3) + 'G')
    # print('Params = ' + str(params / 1000**2) + 'M')