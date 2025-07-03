"""
图形用户界面模块

版权所有 (c) 2025 合肥工业大学。保留所有权利。
作者: 温浩 <2021218526@mail.hfut.edu.cn>
创建日期: 2025-1-21
最后修改: 2025-X-XX
版本: 1.3.0

功能描述:
基于Tkinter实现的图像检索系统用户界面，负责处理用户交互和协调系统工作流。

界面组件:
- 图像选择面板
- 搜索文本输入框
- 结果可视化窗口
- 处理状态指示器

系统架构:
- 类MVC设计模式
- 支持异步处理管道
- 响应式网格布局

技术规格:
- 支持高DPI显示
- 自适应图像缩放
- 跨平台兼容性(Win/macOS/Linux)

交互流程:
1. 用户选择图像文件
2. 输入检索关键词
3. 系统执行分割和检索
4. 可视化显示结果

测试要求:
- 分辨率兼容性测试(1080P/4K)
- 跨平台一致性验证
- 压力测试(>1GB图像文件)

许可证: 专有
"""
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import ImageTk, Image
import cv2
from tqdm import tqdm
from src.models import load_clip_model, load_sam_model
from src.image_utils import *
import os


class ImageRetrievalApp:
    def __init__(self):
        # 初始化设备
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 加载模型
        self.clip_model, self.preprocess = load_clip_model(self.device)

        # 使用绝对路径或相对路径，并检查文件是否存在
        model_path = os.path.join(os.path.dirname(__file__), "../path/model3.pth")
        if os.path.exists(model_path):
            self.sam_model = load_sam_model(model_path)
        else:
            raise FileNotFoundError(f"Model file not found at {model_path}")

        # 初始化GUI
        self.root = tk.Tk()
        self.root.title("Image Segmentation and Retrieval")
        self.root.geometry("900x600")
        self.root.configure(bg="#F0F0F0")

        # 初始化变量
        self.image_path = None

        # 创建界面组件
        self.create_widgets()

    def create_widgets(self):
        # 搜索文本输入
        tk.Label(self.root, text="Enter search text:", bg="#F0F0F0",
                 font=("Helvetica", 14)).grid(row=0, column=0, padx=20, pady=20)
        self.entry = tk.Entry(self.root, font=("Helvetica", 14))
        self.entry.grid(row=0, column=1, padx=20, pady=20)

        # 图片选择按钮
        select_btn = tk.Button(self.root, text="Select Image", command=self.select_image,
                               font=("Helvetica", 14), bg="#4CAF50", fg="white")
        select_btn.grid(row=1, column=0, columnspan=2, pady=20)

        # 图片路径显示
        self.image_label = tk.Label(self.root, text="No image selected",
                                    bg="#F0F0F0", font=("Helvetica", 12))
        self.image_label.grid(row=2, column=0, columnspan=2, pady=20)

        # 图片显示区域
        self.original_label = tk.Label(self.root, bg="#F0F0F0")
        self.original_label.grid(row=3, column=0, padx=20, pady=20, sticky="nsew")

        self.result_label = tk.Label(self.root, bg="#F0F0F0")
        self.result_label.grid(row=3, column=1, padx=20, pady=20, sticky="nsew")

        # 处理按钮
        process_btn = tk.Button(self.root, text="Process Image", command=self.process_image,
                                font=("Helvetica", 14), bg="#008CBA", fg="white")
        process_btn.grid(row=4, column=0, columnspan=2, pady=20)

        # 配置网格行列权重
        self.root.grid_columnconfigure(0, weight=1)
        self.root.grid_columnconfigure(1, weight=1)
        self.root.grid_rowconfigure(3, weight=1)

    def select_image(self):
        self.image_path = filedialog.askopenfilename(
            filetypes=[("Image Files", "*.png;*.jpg;*.jpeg")])
        if self.image_path:
            self.image_label.config(text=f"Selected Image: {self.image_path}")
            self.show_thumbnail(self.original_label, self.image_path)

    def show_thumbnail(self, label, image_path):
        image = Image.open(image_path)
        image.thumbnail((300, 300), Image.LANCZOS)
        photo = ImageTk.PhotoImage(image)
        label.config(image=photo)
        label.image = photo

    def process_image(self):
        search_text = self.entry.get()
        if not search_text:
            messagebox.showwarning("Input Error", "Please enter a search text.")
            return
        if not self.image_path:
            messagebox.showwarning("Input Error", "Please select an image.")
            return

        # 读取并处理图像
        image = cv2.imread(self.image_path)
        if image is None:
            print(f"无法读取图像: {self.image_path}")
            return
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # 生成分割掩码
        masks = self.sam_model.generate(image)

        # 获取所有分割区域图像
        cropped_boxes = []
        for mask in tqdm(masks, desc="Processing masks"):
            segmented_img = segment_image(Image.fromarray(image), mask["segmentation"])
            cropped_box = segmented_img.crop(convert_box_xywh_to_xyxy(mask["bbox"]))
            cropped_boxes.append(cropped_box)

        # 检索相关区域
        scores = retrieve_images(self.clip_model, self.preprocess, self.device,
                                 cropped_boxes, search_text)
        indices = [i for i, v in enumerate(scores) if v > 0.05]

        # 创建并显示结果
        result_image = create_overlay(self.image_path, masks, indices)
        result_image.thumbnail((300, 300), Image.LANCZOS)
        result_photo = ImageTk.PhotoImage(result_image)
        self.result_label.config(image=result_photo)
        self.result_label.image = result_photo

    def run(self):
        self.root.mainloop()
