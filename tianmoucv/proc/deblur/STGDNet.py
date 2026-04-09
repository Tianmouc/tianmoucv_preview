import os
import importlib
import torch
from collections import OrderedDict
from copy import deepcopy
from os import path as osp
from tqdm import tqdm

from tianmoucv.tools import check_url_or_local_path,download_file
from .archs import define_network
import torch.nn as nn


class TianmoucDeblurNet(nn.Module):
    """Base Deblur model for single image deblur."""

    def __init__(self, ckpt_path=None, _optim = False):
        super(TianmoucDeblurNet, self).__init__()

        # define network
        network_cfg = { 'type': 'oneStageRecurrent3', #三支路     #EFNet #RGBDeblur #STDDN
                        'wf': 48,
                        'fuse_before_downsample': True  #如果是rgbdeblur不用tsd，则注释这一行
                      }
        self.net_g = define_network(deepcopy(network_cfg))
        self.net_g.eval()
        self.print_network(self.net_g)


        if ckpt_path is None:
            ckpt_path = 'http://www.tianmouc.cn:38328/index.php/s/7c5EFiQqqdoSwPx/download/nonintegral_net_g_400000.pth'

        status = check_url_or_local_path(ckpt_path)
        
        if status == 1:
            default_file_name = 'deblur.ckpt'
            if not os.path.exists(default_file_name):
                ckpt_path = download_file(url=ckpt_path,file_name=default_file_name)
            else:
                ckpt_path = default_file_name
            print('load finished')
            
        if ckpt_path is not None:
            self.load_network(self.net_g, ckpt_path, True, param_key='params')
        main_version = int(torch.__version__[0])    
        if main_version==2 and _optim:
            print('compiling model for pytorch version>= 2.0.0')
            self.net_g = torch.compile(self.net_g)
            print('compiled!')
            
    def print_network(self, net):
        net = self.get_bare_model(net)
        net_str = str(net)
        net_params = sum(map(lambda x: x.numel(), net.parameters()))


    def get_bare_model(self, net):
        """Get bare model, especially under wrapping with
        DistributedDataParallel or DataParallel.
        """
        return net

    def set_device(self, device):
        self.device = device
        self.net_g = self.net_g.to(device)
        
    def _print_different_keys_loading(self, crt_net, load_net, strict=True):
        crt_net = self.get_bare_model(crt_net)
        crt_net = crt_net.state_dict()
        crt_net_keys = set(crt_net.keys())
        load_net_keys = set(load_net.keys())

    def load_network(self, net, load_path, strict=True, param_key='params'):
        """Load network.

        Args:
            load_path (str): The path of networks to be loaded.
            net (nn.Module): Network.
            strict (bool): Whether strictly loaded.
            param_key (str): The parameter key of loaded network. If set to
                None, use the root 'path'.
                Default: 'params'.
        """
        net = self.get_bare_model(net)
        load_net = torch.load(load_path, map_location='cpu',weights_only=False)
        if param_key is not None:
            load_net = load_net[param_key]
        print(' load net keys', load_net.keys)
        # remove unnecessary 'module.'
        for k, v in deepcopy(load_net).items():
            if k.startswith('module.'):
                load_net[k[7:]] = v
                load_net.pop(k)
        self._print_different_keys_loading(net, load_net, strict)
        net.load_state_dict(load_net, strict=strict)

    def forward(self,data):

        lq = data['frame'].to(self.device)
        td_voxel = data['td_voxel'].to(self.device).unsqueeze(0)  
        sd_voxel = data['sd_voxel'].to(self.device).unsqueeze(0)  

        if len(lq.shape) == 3:
            lq = lq.unsqueeze(0)  #
        if len(td_voxel.shape) == 3:
            td_voxel = td_voxel.unsqueeze(0) 
        if len(sd_voxel.shape) == 3:
            sd_voxel = sd_voxel.unsqueeze(0) 

        assert len(lq.shape) == 4
        assert len(td_voxel.shape) == 4
        assert len(sd_voxel.shape) == 4
    
        pred = self.net_g(x=lq, td=td_voxel, sd=sd_voxel)[0]
    
        return pred

