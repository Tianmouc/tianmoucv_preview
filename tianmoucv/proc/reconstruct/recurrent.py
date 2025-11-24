import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import time

from tianmoucv.tools import check_url_or_local_path,download_file
from tianmoucv.isp import upsampleTSD
from tianmoucv.proc.nn.recurrent_unet import UNetRecurrent
from tianmoucv.proc.nn.utils import spilt_td_batch


def spilt_and_adjust_td_batch(td_):
    '''
    TD+ = ky1 + b = pos
    TD- = y2 = neg
    TD+ = ky2 + b = k*neg+b
    TD- = y1 = (pos-b)/k
    '''
    k = 1#0.7721158189455568 
    b = 0#0.6921550472199398
    
    td_pos = td_.clone()
    td_pos[td_pos<0] = 0
    td_neg = td_.clone()
    td_neg[td_neg>0] = 0
    
    td = torch.cat([td_pos,td_neg],dim=1)

    return td


def normalize_to_01(tensor):
    # 找到当前 tensor 的最小值和最大值
    min_val = tensor.min()
    max_val = tensor.max()

    # 归一化公式：(tensor - min) / (max - min)
    normalized_tensor = (tensor - min_val) / (max_val - min_val + 1e-8)  # 加1e-8避免除以0

    return normalized_tensor
    
class TianmoucRecon_recurrent(nn.Module):
    '''
    重建网络 updated direct td 2024-10-05
    '''
    def __init__(self,ckpt_path =None,_optim=True):
        super(TianmoucRecon_recurrent, self).__init__()
        current_dir=os.path.dirname(__file__)
        
        if ckpt_path is None:
            ckpt_path = 'http://www.tianmouc.cn:38328/index.php/s/Cz4ZHPW7FMj6fXA/download/recurrent2025-08-11_fac_2_extreme_26.0.ckpt'
            
        skip_type='sum'
        recurrent_block_type='convlstm'
        activation='sigmoid'
        num_encoders=4
        base_num_channels=32
        num_residual_blocks=3
        norm= 'IN'
        use_upsample_conv=True
        self.reconNet =  UNetRecurrent(num_input_channels=2,
                                       num_output_channels=1,
                                       skip_type=skip_type,
                                       recurrent_block_type=recurrent_block_type,
                                       activation=activation,
                                       num_encoders=num_encoders,
                                       base_num_channels=base_num_channels,
                                       num_residual_blocks=num_residual_blocks,
                                       norm=norm,
                                       use_upsample_conv=use_upsample_conv)
        
        status = check_url_or_local_path(ckpt_path)
        print('loading..:',ckpt_path)
        if status == 1:
            default_file_name = 'recurrent_best.ckpt'
            if not os.path.exists(default_file_name):
                ckpt_path = download_file(url=ckpt_path,file_name=default_file_name)
            else:
                ckpt_path = default_file_name
            print('load finished')
            
        dict_re = torch.load(ckpt_path, map_location=torch.device('cpu'),weights_only=False)['state_dict_ReconModel']
        dict_reconNet = dict([])
        for key in dict_re:
            new_key_list = key.split('.')[1:]
            new_key = ''
            for e in new_key_list:
                new_key += e + '.'
            new_key = new_key[:-1]
            if 'reconNet' in key:
                dict_reconNet[new_key] = dict_re[key]
        self.reconNet.load_state_dict(dict_reconNet,strict=True)

        self.eval()
        for param in self.reconNet.parameters():
            param.requires_grad = False
        main_version = int(torch.__version__[0])
        if main_version==2 and _optim:
            print('compiling model for pytorch version>= 2.0.0')
            self.reconNet = torch.compile(self.reconNet)
            print('compiled!')


  
    @torch.no_grad() 
    def forward(self, TD, states = None):

        """
        :param x: N x num_input_channels x H x W
        :param prev_states: previous LSTM states for every encoder layer
        :return: N x num_output_channels x H x W
        """
        td = spilt_and_adjust_td_batch(TD)
        f1t_norm = []
        for t in range(td.shape[2]):
            F1t,states = self.reconNet(td[:,:,t,...], states)
            f1t_norm.append(normalize_to_01(F1t))
        f1t_norm = torch.cat(f1t_norm,dim=0)
        
        return f1t_norm, states
    