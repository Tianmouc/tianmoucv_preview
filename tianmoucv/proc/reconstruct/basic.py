import torch
import numpy as np
import cv2
from tianmoucv.proc.nn.utils import tdiff_split,spilt_and_adjust_td_batch

def warp(model,ReconModel,F,td,SD0,SD1,y=None):
    M_list = None
    latent = None   
    embedding_loss1 = 0
    

    if  model == 'unet_original':
        td = td[:,:1,...] - td[:,1:,...]
        Ft,Flow,I_rec,I_warp  = ReconModel(F,td,SD0,SD1,y=y)
        return Ft,Flow,I_rec,I_warp,M_list, embedding_loss1, latent   
        
    if  model == 'unet_mem_v2' or\
        model == 'unet_mem':   
        Ft,Flow,I_rec,I_warp, M_list, embedding_loss1  = ReconModel(F,td,SD0,SD1,y=y)
        return Ft,Flow,I_rec,I_warp,M_list, embedding_loss1, latent

    if  model == 'directtd':                                  
        Ft,Flow,I_rec,I_warp  = ReconModel(F,td)
        return Ft,Flow,I_rec,I_warp,M_list, embedding_loss1, latent  
        
    if  model == 'direct' or \
        model == 'of' or \
        model == 'directUFormer' or \
        model == 'swinir' or \
        model == 'mae' or \
        model == 'of_raft' or \
        model == 'of_raft_tiny' or \
        model == 'direct_mem':  
                                                                  
        Ft,Flow,I_rec,I_warp  = ReconModel(F,td,SD0,SD1)
        return Ft,Flow,I_rec,I_warp,M_list, embedding_loss1, latent  
    
    if model == 'vae':   
        Ft,Flow,I_rec,I_warp, aff, embedding_loss, z_e, mu_state = ReconModel(F,td,SD0,SD1)
        latent = (z_e,mu_state)
        M_list = aff
        return Ft,Flow,I_rec,I_warp, M_list, embedding_loss1, latent
                                                                            
                                                                            
def batch_inference(sample,
                   ReconModel,
                   model='direct',
                   h=320,
                   w=640,
                   device=torch.device('cuda:0'),
                   ifsingleDirection=False,
                   speedUpRate = 1, bs=1,print_info=True): 
    
    with torch.no_grad():
        F0 = sample['F0'].to(device)
        tsdiff = sample['tsdiff'].to(device)
        biasw = (F0.shape[3] - w) // 2
        biash = (F0.shape[2] - h) // 2

        assert w > 0 and biasw >= 0
        assert h > 0 and biash >= 0
        timeLen = tsdiff.shape[2]
        #store results
        Ft_batch = torch.zeros([timeLen,F0.shape[1],h,w]).to(device)
        Ft_reco_batch = torch.zeros([timeLen,F0.shape[1],h,w]).to(device)
        Ft_warp_batch = torch.zeros([timeLen,F0.shape[1],h,w]).to(device)
        batchSize = bs
        tsdiff = tsdiff[:,:,:,biash:h+biash,biasw:w+biasw]
        
        if print_info:
            print('tsdiff 有',timeLen,'帧，重建',timeLen,'帧后舍弃最后1帧')
            print(timeLen,timeLen/batchSize,np.ceil(timeLen/batchSize))
        batch =  int(np.ceil(timeLen/batchSize))
        
        for b in range(batch):
            biast = b * batchSize
            res = min(timeLen-biast,batchSize)
            if res ==0:
                break
            F_batch = torch.zeros([res,F0.shape[1],h,w]).to(device)
            SD0_batch = torch.zeros([res,2,h,w]).to(device)
            SD1_batch = torch.zeros([res,2,h,w]).to(device)
            td_batch = torch.zeros([res,2,h,w]).to(device)
            td_batch_inverse = torch.zeros([res,2,h,w]).to(device)
            #print('processing... ',b, 'batch, reconstruct:',biast+0,'~',biast+res-1)
            for rawt in range(res):#F0->F1-dt
                t = rawt*speedUpRate
                #print('biast~biast+t',0,biast+t)
                SD0_batch[rawt,...] = tsdiff[:,1:,0,...]
                SD1_batch[rawt,...] = tsdiff[:,1:,biast+t,...]
                F_batch[rawt,...] = F0[:,:,biash:h+biash,biasw:w+biasw]
                    
                if t == 0 and b == 0:
                    td_batch[rawt,...] = 0
                else:
                    td_ = tsdiff[:,0:1,1:biast+t+1,...]
                    td,_ = spilt_and_adjust_td_batch(td_)
                    td_batch[rawt,...] = td

            if print_info:        
                print('finished:',biast,'->',biast+res-1,' ALL:',Ft_batch.size(0))        
            Ft1,_,I_1_rec,I_1_warp,M_list, _, latent = warp(model,ReconModel,F_batch, 
                                                                              td_batch, 
                                                                              SD0_batch, 
                                                                              SD1_batch,y=None)
            if not ifsingleDirection and res>=1:
                F1 = sample['F1'].to(device)
                for rawt in range(res):#F0->F1-dt
                    t = rawt*speedUpRate
                    SD0_batch[rawt,...] = tsdiff[:,1:,-1,...]
                    SD1_batch[rawt,...] = tsdiff[:,1:,biast+t,...]
                    F_batch[rawt,...] = F1[:,:,biash:h+biash,biasw:w+biasw]

                    if t+1 == res and b == batch-1:
                        td_batch[rawt,...] = 0
                    else:
                        td_ = tsdiff[:,0:1,biast+t+1:,...]
                        _,td_inverse = spilt_and_adjust_td_batch(td_)
                        td_batch_inverse[rawt,...] = td_inverse

                if print_info:
                    print('finished:',biast+t,'->',-1,' ALL:',Ft_batch.size(0))
                Ft2,_,I_2_rec,I_2_warp,M_list, _, latent  = warp(model,ReconModel,F_batch, 
                                                                              td_batch_inverse, 
                                                                              SD0_batch, 
                                                                              SD1_batch,y=None)
                Ft1 = (Ft1+Ft2)/2
                I_1_rec = (I_1_rec+I_2_rec)/2
                I_1_warp = (I_1_warp+I_2_warp)/2
                
            Ft_batch[biast:biast+res,...] = Ft1.clone()

        Ft_batch = Ft_batch[:-1,...]    
        
        return Ft_batch