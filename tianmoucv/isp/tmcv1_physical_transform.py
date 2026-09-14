import torch
import torch.nn.functional as F

def tmc_to_physical_grid(raw_tsd):
    """
    将原始 [TD, SDL, SDR] 转换为物理对齐的张量空间。
    原始 HxW 映射到 Hx2W (因为每行 SD 分量只占用了 W，两行交替后高度不变，宽度翻倍)
    
    Args:
        raw_tsd: [B, 3, T, H, W] -> [TD, SDL, SDR]
    
    Returns:
        physical_txy: [B, 3, T, H, 2W] -> [TD_grid, SD_lu_dr, SD_dl_ur]
        masks: [B, 3, H, 2W] 标记哪些位置是原始采样点
    """
    B, C, T, H, W = raw_tsd.shape
    device = raw_tsd.device
    
    TD = raw_tsd[:, 0]  # [B, T, H, W]
    SDL = raw_tsd[:, 1]
    SDR = raw_tsd[:, 2]
    
    # 初始化 H x 2W 的张量
    W2 = 2 * W
    sd_lu_dr = torch.zeros((B, T, H, W2), device=device)
    sd_dl_ur = torch.zeros((B, T, H, W2), device=device)
    td_grid = torch.zeros((B, T, H, W2), device=device)
    
    # 建立 mask
    mask_lu_dr = torch.zeros((B, 1, H, W2), device=device)
    mask_dl_ur = torch.zeros((B, 1, H, W2), device=device)
    mask_td = torch.zeros((B, 1, H, W2), device=device)
    
    # 偶数行 (y=0, 2, 4...)
    # SDul (SDL_even) -> [y, 2x]
    sd_lu_dr[:, :, 0::2, 0::2] = SDL[:, :, 0::2, :]
    mask_lu_dr[:, :, 0::2, 0::2] = 1.0
    
    # SDur (SDR_even) -> [y, 2x+1]
    sd_dl_ur[:, :, 0::2, 1::2] = SDR[:, :, 0::2, :]
    mask_dl_ur[:, :, 0::2, 1::2] = 1.0
    
    # TD_even -> [y, 2x]
    td_grid[:, :, 0::2, 0::2] = TD[:, :, 0::2, :]
    mask_td[:, :, 0::2, 0::2] = 1.0
    
    # 奇数行 (y=1, 3, 5...)
    # -SDdr (-SDR_odd) -> [y, 2x+1]
    sd_lu_dr[:, :, 1::2, 1::2] = -SDR[:, :, 1::2, :]
    mask_lu_dr[:, :, 1::2, 1::2] = 1.0
    
    # -SDdl (-SDL_odd) -> [y, 2x]
    sd_dl_ur[:, :, 1::2, 0::2] = -SDL[:, :, 1::2, :]
    mask_dl_ur[:, :, 1::2, 0::2] = 1.0
    
    # TD_odd -> [y, 2x+1]
    td_grid[:, :, 1::2, 1::2] = TD[:, :, 1::2, :]
    mask_td[:, :, 1::2, 1::2] = 1.0
    
    # 4邻域插值填充
    def fill_holes(x, mask):
        kernel = torch.tensor([[0, 1, 0], [1, 0, 1], [0, 1, 0]], device=device, dtype=torch.float32).view(1, 1, 3, 3)
        # x shape: [B, T, H, W2]
        x_flat = x.view(-1, 1, H, W2)
        m_flat = mask.repeat(1, T, 1, 1).view(-1, 1, H, W2)
        
        sum_neighbors = F.conv2d(x_flat, kernel, padding=1)
        count_neighbors = F.conv2d(m_flat, kernel, padding=1)
        
        interpolated = sum_neighbors / (count_neighbors + 1e-8)
        result = x_flat * m_flat + interpolated * (1 - m_flat)
        return result.view(B, T, H, W2)

    sd_lu_dr = fill_holes(sd_lu_dr, mask_lu_dr)
    sd_dl_ur = fill_holes(sd_dl_ur, mask_dl_ur)
    td_grid = fill_holes(td_grid, mask_td)
    
    physical_txy = torch.stack([td_grid, sd_lu_dr, sd_dl_ur], dim=1) # [B, 3, T, H, 2W]
    masks = torch.cat([mask_td, mask_lu_dr, mask_dl_ur], dim=1)      # [B, 3, H, 2W]
    
    return physical_txy, masks

def physical_grid_to_tmc(sd_lu_dr, sd_dl_ur, td_grid, H, W):
    """
    物理空间回到原始 [TD, SDL, SDR] 空间。
    """
    B, T, H_cur, W_cur = sd_lu_dr.shape
    device = sd_lu_dr.device
    
    rawdiff = torch.zeros((B, 3, T, H, W), device=device)
    
    # 偶数行
    rawdiff[:, 0, :, 0::2, :] = td_grid[:, :, 0::2, 0::2] # TD
    rawdiff[:, 1, :, 0::2, :] = sd_lu_dr[:, :, 0::2, 0::2] # SDL
    rawdiff[:, 2, :, 0::2, :] = sd_dl_ur[:, :, 0::2, 1::2] # SDR
    
    # 奇数行
    rawdiff[:, 0, :, 1::2, :] = td_grid[:, :, 1::2, 1::2] # TD
    rawdiff[:, 1, :, 1::2, :] = -sd_dl_ur[:, :, 1::2, 0::2] # SDL
    rawdiff[:, 2, :, 1::2, :] = -sd_lu_dr[:, :, 1::2, 1::2] # SDR
    
    return rawdiff
