# TianmouCV Introduction

这一版 `introduction` 目录是一条从入门到进阶的教程主线，目标很明确：

- 让**从未接触过 Tianmouc** 的用户也能顺着跑一遍
- 跑完之后知道 TianmouCV 的核心能力边界
- 知道常用 API 在哪里、该怎么组合
- 知道后续高级算法和工程示例该从哪里继续

## 这套教程适合谁

- 第一次接触 Tianmouc 传感器与 TianmouCV
- 想快速了解 `data / isp / sim / proc` 这些模块分别负责什么
- 想先用 CPU 跑通一套完整流程，再逐步进入大模型、硬件和实时系统

## 设计原则

- 默认数据统一使用官方分享的测试样本：`fishe8.7z`
- 默认走 CPU 友好的路径，避免第一步就被 GPU / 权重 / 硬件阻塞
- Notebook 中直接调用 TianmouCV 主要 API，不用教程函数再次封装核心能力
- `tianmoucv.utils` 只保留与 TianmouCV 自身 API 无关的通用辅助，例如测试数据下载、样本枚举、数组归一化、视频保存

## 使用建议

- 如果你是第一次上手：从 `Step 0` 开始按顺序跑
- 如果你已经会读数据：可以直接从 `Step 3` 或 `Step 4` 开始
- 如果你只关心下游算法：建议至少先跑完 `Step 0`、`Step 1`、`Step 3`
- 如果你只想看训练链路：可以直接跳到 `Step 9`

## 先决条件

- 推荐在仓库根目录对应的 Python 环境中运行 Notebook
- 自动下载测试数据依赖 `requests`
- 自动解压 7z 依赖以下任一条件：
  - 系统存在 `bsdtar` 或 `7z`
  - 或 Python 环境中安装 `py7zr`

## 你会得到什么

按顺序跑完这套教程后，应该能完成下面这些事情：

- 自动下载并读取 Tianmouc 官方测试样本
- 理解 `F0 / F1 / rawDiff / tsdiff / txydiff / meta` 的含义
- 熟悉 `TianmoucDataReader`、decoder、parser 的不同定位
- 掌握 `default_rgb_isp`、`vizDiff`、`SD2XY` 等高频 API
- 跑通经典重建、经典光流、差分特征跟踪
- 看懂 TianmouCV 的高级模型地图
- 用默认测试数据跑一次 `tmc-gray-recon` 的训练 smoke test

## 推荐学习路径

| Step | Notebook | 目标 | 跑完后你会得到什么 | 关键 API |
|---|---|---|---|---|
| 0 | `00_setup_and_test_data.ipynb` | 自动下载测试数据并验证读取环境 | 一套可复用的默认测试数据与 reader 环境 | `ensure_tutorial_dataset`, `TianmoucDataReader` |
| 1 | `01_quickstart_data_reader.ipynb` | 认识 sample 结构 | 看懂 Tianmouc 一条 sample 的时间、空间、合成与尺寸关系 | `TianmoucDataReader`, `rawDiff`, `tsdiff`, `txydiff`, `SD2XY`, `vizDiff` |
| 2 | `02_parser_and_decoders.ipynb` | 理解 parser 与单路径解码器 | 知道高层 reader 和底层 decoder 各自适合什么场景 | `TianmoucDataSampleParser`, `cone_tmdat_to_npy`, `rod_tmdat_to_npy` |
| 3 | `03_visualization_and_isp.ipynb` | 掌握最常用 ISP 与可视化 | 能独立完成 RGB/AOP 的基础可视化与信号变换 | `default_rgb_isp`, `gray_world_awb`, `ACESToneMapping`, `vizDiff`, `SD2XY` |
| 4 | `04_simulator.ipynb` | 用仿真器理解 AOP/COP 关系 | 对仿真输入输出和差分信号形成直觉 | `run_sim_singleimg`, `visualize_diff` |
| 5 | `05_aop_denoise_and_tsd_transform.ipynb` | 体验 AOP 去噪与几何变换 | 能对 AOP 做去噪、翻转、旋转、physical grid 变换 | `denoise_defualt_args`, `flip_xyd`, `rotate_txyd`, `tmc_to_physical_grid` |
| 6 | `06_reconstruction_basics.ipynb` | 跑通经典重建算法 | 知道 TD / SD / 泊松融合三类经典重建入口 | `TD_integration`, `SD_integration`, `poisson_blending` |
| 7 | `07_optical_flow_and_tracking.ipynb` | 体验运动分析任务 | 能基于 AOP 跑经典光流与特征跟踪 | `LK_optical_flow`, `HS_optical_flow`, `Feature_tracker_sd` |
| 8 | `08_advanced_models_overview.ipynb` | 建立高级模型地图 | 知道学习型重建、光流、分割、去模糊对应哪些入口和 example | `TianmoucRecon_*`, `TianmoucOF_*`, `TM_seg` |
| 9 | `09_training_quickstart_gray_recon.ipynb` | 用默认测试数据做一次训练 smoke test | 跑通一次最小训练链路，建立训练全流程直觉 | `TianmoucDataReader`, `tmc-gray-recon` |

## 当前保留的公共辅助工具

这些工具只处理教程外围事务，不封装 TianmouCV 核心 API：

- `ensure_tutorial_dataset()`：自动下载并解压 `fishe8.7z`
- `list_available_samples()`：列出含 `cone/` 与 `rod/` 的样本目录
- `normalize_to_uint8()` / `ensure_uint8_image()`：把数组转成可显示/可保存的 `uint8` 图像
- `save_video()`：把帧序列保存为视频

## 进阶方向

如果你已经跑完 `Step 0` 到 `Step 7`，可以继续按兴趣深入：

- 重建方向：`reconstruct_tiny_unet.ipynb`, `reconstruct_fuse_net.ipynb`, `reconstruct_recurrent.ipynb`
- 光流方向：`opticalflow_spynet.ipynb`, `opticalflow_RAFT.ipynb`
- 去噪 / 去模糊：`denoise_tmdat_lvatf.ipynb`, `deblur_stgdnet.ipynb`
- 检测 / 分割：`segmentation.ipynb`
- 实时 / 硬件：`open_camera.py`, `realtime_inf.py`, `qrcode_demo/qrcode.py`, `calibration_OpenCV.ipynb`

## 一句话导航

- 想快速知道 TianmouCV 怎么读数据：看 `Step 0-2`
- 想快速知道 TianmouCV 能做哪些基础处理：看 `Step 3-5`
- 想快速知道 TianmouCV 能做哪些经典任务：看 `Step 6-7`
- 想快速知道 TianmouCV 的高级模型版图：看 `Step 8`
- 想快速知道训练链路怎么搭起来：看 `Step 9`
