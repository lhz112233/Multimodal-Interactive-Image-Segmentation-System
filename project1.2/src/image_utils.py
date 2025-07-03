"""
计算机视觉工具模块

版权所有 (c) 2025 合肥工业大学。保留所有权利。
作者: 温浩 <2021218526@mail.hfut.edu.cn>
创建日期: 2025-1-21
最后修改: 2025-X-XX
版本: 1.1.0

功能描述:
提供图像检索系统的核心计算机视觉操作，包括图像分割、特征提取和可视化叠加生成。

主要功能:
- 边界框坐标系统转换
- 带背景移除的图像分割
- 基于CLIP的图文相似度计算
- 透明叠加层合成

关键函数:
- convert_box_xywh_to_xyxy(): 坐标系统转换
- segment_image(): 背景移除图像分割
- retrieve_images(): 语义相似度计算
- create_overlay(): 可视化叠加生成

性能优化:
- GPU加速支持
- 内存高效的张量运算
- 批量处理支持

许可证: 专有
"""
import cv2
import numpy as np
from PIL import Image, ImageDraw
import torch
import clip


def convert_box_xywh_to_xyxy(box):
    return [box[0], box[1], box[0] + box[2], box[1] + box[3]]


def segment_image(image, segmentation_mask):
    image_array = np.array(image)
    segmented_image_array = np.zeros_like(image_array)
    segmented_image_array[segmentation_mask] = image_array[segmentation_mask]
    segmented_image = Image.fromarray(segmented_image_array)

    black_image = Image.new("RGB", image.size, (0, 0, 0))
    transparency_mask = Image.fromarray(segmentation_mask.astype(np.uint8) * 255)
    black_image.paste(segmented_image, mask=transparency_mask)
    return black_image


@torch.no_grad()
def retrieve_images(model, preprocess, device, elements: list, search_text: str):
    preprocessed_images = [preprocess(image).to(device) for image in elements]
    tokenized_text = clip.tokenize([search_text]).to(device)
    stacked_images = torch.stack(preprocessed_images)

    image_features = model.encode_image(stacked_images)
    text_features = model.encode_text(tokenized_text)

    image_features /= image_features.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)

    probs = 100.0 * (image_features @ text_features.T)
    return probs[:, 0].softmax(dim=0)


def create_overlay(image_path, masks, indices):
    original_image = Image.open(image_path).convert("RGBA")
    overlay_image = Image.new("RGBA", original_image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay_image)

    for seg_idx in indices:
        mask = masks[seg_idx]["segmentation"]
        mask_image = Image.fromarray(mask.astype(np.uint8) * 255)
        draw.bitmap((0, 0), mask_image, fill=(255, 0, 0, 200))

    return Image.alpha_composite(original_image, overlay_image)