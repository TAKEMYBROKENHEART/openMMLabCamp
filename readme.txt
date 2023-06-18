# 结果
+------------+-------+-------+
|   Class    |  IoU  |  Acc  |
+------------+-------+-------+
|    red     | 80.24 |  98.5 |
|   green    | 79.55 | 83.38 |
|   white    | 20.92 | 22.03 |
| seed-black | 35.64 | 35.75 |
| seed-white | 56.02 |  80.0 |
|  unnamed   |  0.67 |  0.67 |
+------------+-------+-------+

# 下载数据集
wget https://zihao-openmmlab.obs.cn-east-3.myhuaweicloud.com/20230130-mmseg/dataset/watermelon/Watermelon87_Semantic_Seg_Mask.zip
# 解压
unzip Watermelon87_Semantic_Seg_Mask.zip >> /dev/null # 解压
# 下载mmsegmentation
git clone https://github.com/open-mmlab/mmsegmentation.git
cd mmsegmentation
# 数据处理
'''
import os
import cv2
img_root ="Watermelon87_Semantic_Seg_Mask/img_dir/train/"
for path in os.listdir(img_dir):
    #打开图片
    img = cv2.imread(os.path.join(img_root,path))

    #改名字写进去
    if path.endswith('png') or path.endswith('jpeg'):
        new_name = path.split(".")[0]+".jpg"
        cv2.imwrite(os.path.join(img_root,new_name),img)
        #删除原图片
        os.remove(os.path.join(img_root,path))
'''
# 准备配置文件
'''
import numpy as np
import os.path as osp
from tqdm import tqdm
import mmcv
import mmengine

#定义数据集
!rm -rf mmseg/datasets/DubaiDataset.py # 删除原有文件
!wget https://zihao-openmmlab.obs.cn-east-3.myhuaweicloud.com/20230130-mmseg/Dubai/DubaiDataset.py -P mmseg/datasets

#注册数据集
!rm -rf mmseg/datasets/__init__.py # 删除原有文件
!wget https://zihao-openmmlab.obs.cn-east-3.myhuaweicloud.com/20230130-mmseg/Dubai/__init__.py -P mmseg/datasets

#定义训练和测试pipeline
!rm -rf configs/_base_/datasets/DubaiDataset_pipeline.py
!wget https://zihao-openmmlab.obs.cn-east-3.myhuaweicloud.com/20230130-mmseg/Dubai/DubaiDataset_pipeline.py -P configs/_base_/datasets

#download configure file
!rm -rf configs/pspnet/pspnet_r50-d8_4xb2-40k_DubaiDataset.py # 删除原有文件
!wget https://zihao-openmmlab.obs.cn-east-3.myhuaweicloud.com/20230130-mmseg/Dubai/pspnet_r50-d8_4xb2-40k_DubaiDataset.py -P configs/pspnet 

# load Config
from mmengine import Config
cfg = Config.fromfile('./configs/pspnet/pspnet_r50-d8_4xb2-40k_DubaiDataset.py')

#adjust cfg
cfg.norm_cfg = dict(type='BN', requires_grad=True) # 只使用GPU时，BN取代SyncBN
cfg.crop_size = (256, 256)
cfg.model.data_preprocessor.size = cfg.crop_size
cfg.model.backbone.norm_cfg = cfg.norm_cfg
cfg.model.decode_head.norm_cfg = cfg.norm_cfg
cfg.model.auxiliary_head.norm_cfg = cfg.norm_cfg
# modify num classes of the model in decode/auxiliary head

# 模型 decode/auxiliary 输出头，指定为类别个数
cfg.model.decode_head.num_classes = 6
cfg.model.auxiliary_head.num_classes = 6

cfg.train_dataloader.batch_size = 8

cfg.test_dataloader = cfg.val_dataloader

# 结果保存目录
cfg.work_dir = './work_dirs/Watermelon87_Semantic_Seg_Mask'

# 训练迭代次数
cfg.train_cfg.max_iters = 3000
# 评估模型间隔
cfg.train_cfg.val_interval = 400
# 日志记录间隔
cfg.default_hooks.logger.interval = 100
# 模型权重保存间隔
cfg.default_hooks.checkpoint.interval = 1500

# 随机数种子
cfg['randomness'] = dict(seed=0)
cfg.dump('pspnet-DubaiDataset_20230612.py')
'''

# 修改相关文件
打开pspnet-DubaiDataset_20230612.py

data_root以及train_dataloader、val_dataloader、test_dataloader中的data_root改为'Watermelon87_Semantic_Seg_Mask/'

打开mmseg/datasets/DubaiDataset.py

将DubaiDataset类下METAINFO改为
'''
METAINFO = {
        'classes':['red', 'green','white','seed-black','seed-white','unnamed'],
        'palette':[[132,41,246], [228,193,110], [152,16,60], [58,221,254], [41,169,226], [155,155,155]]
'''

#开始训练

from mmengine.runner import Runner
from mmseg.utils import register_all_modules

# register all modules in mmseg into the registries
# do not init the default scope here because it will be init in the runner
register_all_modules(init_default_scope=False)
runner = Runner.from_cfg(cfg)

runner.train()
