## 0.4.0.0起 更新日志

- 从0.4.0.0 版本起，维护更新日志
- 从0.4.0.0 版本起，增加tianmouc的仿真器模块

1. data.tianmoucData

- 对sample增加了txydiff关键字，将SD转成xy方向输出，可以用于更精确的物理保真的上采样
- datareader的其他关键词逻辑不变，以保持对旧版本兼容

1. isp.transform

- 增加了几个不同的SD空洞填补策略
- 增加了SD（矢量场数据）的旋转和镜像逻辑
- example放在tianmoucv\_example/data/rotate\_tsd.ipynb

1. sim

- 增加了简单的仿真器调用接口，支持图像序列直接转成tianmouc数据
- 输入为按照文件名排序的图像序列
  example放在tianmoucv/tianmoucv\_example/simulator

1. tianmoucv exmaple

- 对应接口更新

## 0.4.0.1 log

- txydiff关键字按照开关输出
- tianmoucv支持pip直接安装

## 0.4.0.2 log

- datareader默认改回tsdiff输出
- sd2xy接口矫正一个scaling系数
- 更新一些readme和下载资源

## 0.4.0.3 log

- cpp不再输出底层debug信息
- single读出模式的一些改善

## 0.4.0.4 log

- 支持一次性读取N帧
- data reader底层读取方法重构

## 0.4.0.5 log

- TD2VID训练，权重下载链接更新，demo更新
- pytorch load权重接口增加一个参数以支持特别新版的pytorch

## 0.4.1.0 log

- 加入数据仿真模块
- requirments严谨
- 加入相机标定模块

## 0.4.1.1 log

- open camera脚本的严重bug，和编译器相关，可能随机造成handle错误，已修
- 文档更新
- 中文文档
- optimize speed an streaming logic of ./camera module on low-end device

## 0.4.1.2 log @25.11.21

- 更新特征追踪器 tianmoucv.proc.tracking.Feature\_tracker\_sd
- 更新去模糊的新算法和权重

## 0.4.1.4 log @26.03.23

- 更新仿真器的增强逻辑
- 数据读取example的优化
- 增加nn加速的上采样

## 0.4.2.0 log @26.04.09

- TianmoucV1仿真器逻辑更新
- **开始引入AI agent**使用AI对项目进行了优化重构和文档编写

## 0.4.2.1 log @26.04.17

- 增加光流算法SD-RAFT  tianmoucv_example/proc/optical_flow/opticalflow_RAFT.ipynb
- 增加实例分割算法 tianmoucv/tianmoucv_example/proc/segmentation.ipynb
- 增加IGFNet视频重建算法 tianmoucv_example/proc/reconstructor/reconstruct_fuse_net.ipynb
