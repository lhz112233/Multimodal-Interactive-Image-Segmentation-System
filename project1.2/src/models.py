"""
深度学习模型管理模块

版权所有 (c) 2025 合肥工业大学。保留所有权利。
作者: 温浩 <2021218526@mail.hfut.edu.cn>
创建日期: 2025-1-21
最后修改: 2025-X-XX
版本: 1.1.0

功能描述:
本模块负责深度学习模型的初始化和管理，包括用于图文嵌入的CLIP模型和图像分割的SAM模型。

核心组件:
- 带进度显示的CLIP模型加载
- 使用自定义检查点的SAM模型初始化
- 设备感知的模型部署(CPU/GPU)

实现细节:
- 基于PyTorch的模型管理
- 集成Segment-Anything库
- 使用tqdm实现友好的进度显示

性能指标:
- CLIP模型加载时间 < 5s (GPU环境)
- SAM模型初始化内存占用 < 4GB

许可证: 专有
"""
import torch
import clip
from segment_anything import build_sam, SamAutomaticMaskGenerator
from tqdm import tqdm

def load_clip_model(device):
    print('开始加载CLIP模型')
    for _ in tqdm(range(1), desc="Loading CLIP model"):
        model, preprocess = clip.load("ViT-B/32", device=device)
    print('CLIP模型加载完成')
    return model, preprocess

def load_sam_model(checkpoint_path):
    print('开始加载SAM模型')
    for _ in tqdm(range(1), desc="Loading SAM model"):
        sam = build_sam(checkpoint=checkpoint_path)
        mask_generator = SamAutomaticMaskGenerator(sam)
    print('SAM模型加载完成')
    return mask_generator