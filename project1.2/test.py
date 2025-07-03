"""
图像检索系统主入口模块

版权所有 (c) 2025 合肥工业大学。保留所有权利。
作者: 温浩 <2021218526@mail.hfut.edu.cn>
创建日期: 2025-1-21
最后修改: 2025-X-XX
版本: 1.1.0

功能描述:
本模块作为图像检索系统的主入口点，负责初始化并启动图形用户界面应用程序。

主要功能:
- 应用程序生命周期管理
- 依赖项初始化

依赖库:
- PyTorch 2.0+
- OpenCV 4.7+
- Pillow 10.0+
- Tkinter 8.6+
- Segment-Anything 1.0+

许可证: 专有
"""
# from src.gui import ImageRetrievalApp

# if __name__ == "__main__":
#     ImageRetrievalApp().run()

# import gradio as gr
# print(gr.__version__)

# import torch
# print(f"PyTorch 版本: {torch.__version__}")
# print(f"CUDA 可用: {torch.cuda.is_available()}")
# print(f"当前设备: {torch.cuda.get_device_name(torch.cuda.current_device())}")
# # 升级 TorchVision（确保与 PyTorch 2.4.1 兼容）


# import torch
# import torchvision
# print(f"PyTorch版本: {torch.__version__}")
# print(f"TorchVision版本: {torchvision.__version__}")

import torch
import clip
from segment_anything import build_sam, SamAutomaticMaskGenerator
from PIL import Image
import numpy as np

# 初始化设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 加载模型
clip_model, preprocess = clip.load("ViT-B/32", device=device)
sam = build_sam(checkpoint="path/model3.pth").to(device)
mask_generator = SamAutomaticMaskGenerator(sam)

# 创建测试图像
image = Image.new('RGB', (224, 224), (255, 255, 255))
image_np = np.array(image)

# 生成掩码
masks = mask_generator.generate(image_np)

# 编码图像
image_tensor = preprocess(image).unsqueeze(0).to(device)
with torch.no_grad():
    features = clip_model.encode_image(image_tensor)

print("测试完成，GPU应短暂被占用。")
