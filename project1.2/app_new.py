import gradio as gr
import torch
import clip
from segment_anything import build_sam, SamAutomaticMaskGenerator
import numpy as np
from PIL import Image, ImageDraw
import os
from datetime import datetime
import pandas as pd
import requests
import threading
import tempfile
import matplotlib.pyplot as plt


# ========== 全局变量 ==========
stop_event = threading.Event()
history_records = []
upload_mode = ""

# ========== 初始化模块 ==========
# 初始化设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"当前设备: {torch.cuda.get_device_name(torch.cuda.current_device())}")
# 加载分割模型
def load_segmentation_models():
    print('开始加载CLIP模型...')
    clip_model, preprocess = clip.load("ViT-B/32", device=device)
    print('CLIP模型加载完成,设备：', device, ' 模型：', 'ViT-B/32')    
    print('开始加载SAM模型...')
    model_path = os.path.join(os.path.dirname(__file__), "path/model3.pth")
    if os.path.exists(model_path):
        sam = build_sam(checkpoint=model_path)
        # sam.to(device)
        mask_generator = SamAutomaticMaskGenerator(sam)
        print('SAM模型加载完成')
        return clip_model, preprocess, mask_generator
    else:
        raise FileNotFoundError(f"Model file not found at {model_path}")

clip_model, preprocess, mask_generator = load_segmentation_models()

# ========== 图像处理模块 ==========
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
def retrieve_images(elements: list, search_terms: list, process_info):
    start_time = datetime.now()

    # 预处理所有图像
    preprocessed_images = [preprocess(img).to(device) for img in elements]
    stacked_images = torch.stack(preprocessed_images)

    # 编码文本
    tokenized_text = clip.tokenize(search_terms).to(device)

    # 提取特征
    image_features = clip_model.encode_image(stacked_images)
    text_features = clip_model.encode_text(tokenized_text)

    # 归一化特征
    image_features /= image_features.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)

    # 计算相似度并取最大值
    probs = 100.0 * (image_features @ text_features.T)
    max_probs, _ = torch.max(probs, dim=1)
    scores = max_probs.softmax(dim=0)

    process_info["clip_time"] = (datetime.now() - start_time).total_seconds()
    return scores

def createOverlay(image, masks, indices):
    original_image = image.convert("RGBA")
    overlay_image = Image.new("RGBA", original_image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay_image)
    for seg_idx in indices:
        mask = masks[seg_idx]["segmentation"]
        mask_image = Image.fromarray(mask.astype(np.uint8) * 255)
        draw.bitmap((0, 0), mask_image, fill=(240, 135, 132, 200))
    return Image.alpha_composite(original_image, overlay_image)

def save_temp_file(image: Image.Image):
    temp_dir = tempfile.mkdtemp()
    temp_path = os.path.join(temp_dir, "temp_image.png")
    image.save(temp_path)
    return temp_path

def process_image_batch(input_images, input_webcam, upload_mode, search_text, show_mask, threshold, box_nms_thresh):    
    stop_event.clear()
    process_info = {"timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
    mask_generator.box_nms_thresh = box_nms_thresh
    
    # 统一处理图像源
    if upload_mode == "本地上传":
        image_list = input_images
    else:
        image_list = [save_temp_file(input_webcam)]
    
    # 分割搜索关键词
    search_terms = [term.strip() for term in search_text.split(',') if term.strip()]
    if not search_terms:
        return [None] * len(image_list), pd.DataFrame(), "未输入有效搜索关键词"

    print(f"upload_mode: {upload_mode}, image_list: {image_list}, search_text: {search_text}")

    result_images = []
    all_logs = []
    
    for idx, image_file in enumerate(image_list, 1):
        if stop_event.is_set():
            all_logs.append("处理已由用户终止")
            break
        
        try:
            # 直接打开上传的图片文件
            # image = Image.open(image_file) if isinstance(image_file, str) else image_file
            image = Image.open(image_file)
            image_rgb = image.convert("RGB")
            image_np = np.array(image_rgb)
            
            # 生成分割掩码
            masks = mask_generator.generate(image_np)
            cropped_boxes = [segment_image(image_rgb, m["segmentation"]).crop(convert_box_xywh_to_xyxy(m["bbox"]))
                            for m in masks]
            
            # 检索匹配区域
            scores = retrieve_images(cropped_boxes, search_terms, process_info)
            indices = [i for i, v in enumerate(scores) if v > threshold]

            # 生成结果图像
            result_img = createOverlay(image_rgb, masks, indices) if show_mask else image_rgb
            result_images.append(result_img)

            log = f"图片{idx}: 检测到{len(indices)}个匹配区域 | 耗时{process_info['clip_time']:.2f}s"
            all_logs.append(log)
            
            # 更新历史记录
            global history_records
            history_records.append([
                process_info["timestamp"], search_text, threshold, len(indices), process_info["clip_time"]
            ])

        except Exception as e:
            all_logs.append(f"图片{idx}处理失败: {str(e)}")
            result_images.append(None)
    
    log_df = pd.DataFrame({"处理日志": all_logs})
    stats = f"共处理{len(result_images)}张图片 | 总耗时{sum([process_info.get('clip_time', 0)]):.2f}秒" 
    history_df = pd.DataFrame(history_records, columns=["时间", "搜索内容", "阈值", "匹配数", "耗时"])   
    
    return result_images, log_df, stats, history_df

def stop_processing():
    stop_event.set()

# ========== 界面模块 ==========
# 修改图像分割界面创建函数
def create_image_segmentation_interface():
    with gr.Blocks() as interface:
        with gr.Row():
            input_col = gr.Column(scale=3)
            output_col = gr.Column(scale=7)
            with input_col:
                with gr.Group():
                    # 新增上传模式选择
                    upload_mode = gr.Radio(
                        choices=["本地上传", "拍照上传"],
                        label="上传模式",
                        value="本地上传"
                    )
                    
                    # 本地上传组件
                    input_images = gr.Files(label="上传图片", file_types=["image"], file_count="multiple")
                    # 拍照上传组件
                    input_webcam = gr.Image(label="拍照上传", sources="webcam", type="pil", show_download_button=True, visible=False)
                    # 画廊组件
                    input_gallery = gr.Gallery(label="上传图片预览", columns=3, height=300)
                    # 动态切换上传组件
                    def toggle_upload_mode(mode):
                        return {
                            input_images: gr.update(visible=mode == "本地上传"),
                            input_webcam: gr.update(visible=mode == "拍照上传"),
                        }
                    
                    upload_mode.change(
                        fn=toggle_upload_mode,
                        inputs=upload_mode,
                        outputs=[input_images, input_webcam]
                    )

                    # 预览上传图片
                    input_images.change(
                        fn=lambda files: [Image.open(f) for f in files] if files else [],
                        inputs=input_images,
                        outputs=input_gallery
                    )
                    
                    input_webcam.change(
                        fn=lambda img: [img] if img is not None else [],
                        inputs=input_webcam,
                        outputs=input_gallery
                    )
                    
                    # 添加示例提示词
                    example_prompts = [
                        ["horse, fish"],
                        ["dog, cat"],
                        ["car, bicycle, airplane"],
                        ["person, building, sky"],
                        ["flower, tree, mountain"],
                        ["sun, moon, stars"],
                        ["food, drink, clothes"],
                        ["fruit, vegetable, animal"],
                    ]
                    
                   
                    search_text = gr.Textbox(label="搜索内容", placeholder="eg: person, dog, car, ...")
                    
                    # 添加点击示例
                    gr.Examples(
                        label="搜索模板",
                        examples=example_prompts,
                        inputs=[search_text],
                        examples_per_page=3,
                    )
                    
                    with gr.Row():
                        threshold = gr.Slider(0.01, 0.5, value=0.1, label="匹配阈值")
                        box_nms_thresh = gr.Slider(0.1, 0.9, value=0.7, label="区域合并阈值")
                        
                    
                    stop_btn = gr.Button("🛑 终止处理", variant="stop")
                    process_btn = gr.Button("🚀 开始处理", variant="primary")

            with output_col:
                with gr.Tabs():
                    with gr.Tab("结果展示"):
                        output_gallery = gr.Gallery(label="处理结果", columns=3, height=800)
                    with gr.Tab("处理日志"):
                        logs = gr.Dataframe(label="日志记录", headers=["处理详情"])
                        stats = gr.Textbox(label="统计信息")
                    with gr.Tab("历史记录"):
                        history_table = gr.Dataframe(
                            headers=["时间", "搜索内容", "阈值", "匹配数", "耗时"],
                            datatype=["str", "str", "number", "number", "number"],
                            interactive=False,
                        height=400
                        )
                        # 清除历史按钮
                        clear_history_btn = gr.Button("🗑 清除历史记录", variant="secondary")
                        # 绑定清除功能
                        def clear_history():
                            global history_records
                            history_records = []
                            return pd.DataFrame(columns=["时间", "搜索内容", "阈值", "匹配数", "耗时"])
      
        clear_history_btn.click(
            fn=clear_history,
            inputs=None,
            outputs=history_table
        )
            
        process_btn.click(
            fn=process_image_batch,
            inputs=[
                input_images,    # 图像源1
                input_webcam,    # 图像源2
                upload_mode,     # 上传模式
                search_text,     # 搜索内容
                gr.Checkbox(True, visible=False),# 强制显示掩码结果 
                threshold,       # 匹配阈值
                box_nms_thresh   # 区域合并阈值
            ],
            outputs=[output_gallery, logs, stats, history_table]
        )
        
        # if upload_mode == "本地上传":
        #     process_btn.click(
        #     fn=process_image_batch,
        #     inputs=[
        #         input_images,    # 图像源
        #         upload_mode,     # 上传模式
        #         search_text,     # 搜索内容
        #         gr.Checkbox(True, visible=False),# 强制显示掩码结果 
        #         threshold,       # 匹配阈值
        #         box_nms_thresh   # 区域合并阈值
        #     ],
        #     outputs=[output_gallery, logs, stats, history_table]
        # )
        # else:  # 拍照上传模式
        #     process_btn.click(
        #     fn=process_image_batch,
        #     inputs=[
        #         input_webcam,    # 图像源
        #         upload_mode,     # 上传模式
        #         search_text,     # 搜索内容
        #         gr.Checkbox(True, visible=False),# 强制显示掩码结果 
        #         threshold,       # 匹配阈值
        #         box_nms_thresh   # 区域合并阈值
        #     ],
        #     outputs=[output_gallery, logs, stats, history_table]
        # )
        
        
        stop_btn.click(fn=lambda: stop_event.set(), inputs=None, outputs=None)
        # stop_btn.click(
        #     fn=process_image_batch,
        #     inputs=[],
        #     outputs=[]
        # )
        
    return interface

# ========== 聊天模块 ==========
DEEPSEEK_API_KEY = "sk-03b1f58d372240078828fc3c47482ca6"
API_URL = "https://api.deepseek.com/v1/chat/completions"

def doChatbot(message, history):
    headers = {"Authorization": f"Bearer {DEEPSEEK_API_KEY}", "Content-Type": "application/json"}
    messages = [{"role": "system", "content": "你是一个集成化多模态图像分析系统 已完成核心算法与全栈交互闭环的构建 1 通过融合SAM模型的精准分割能力与CLIP的跨模态语义理解 实现物体级自动分割 多标签分类及动态掩码可视化 结合异步批处理技术达到高效并行运算 2 基于DeepSeek大模型构建上下文感知型AI助手 支持参数调优建议与结果智能解析 配合滑动式阈值调节和NMS合并系数配置面板形成自适应工作流 3 采用Gradio框架打造三模态操作界面 集成动态缩略图预览 双维度日志追踪及分割-检索-对话协同工作区 实现上传-分析-交互的全流程可视化闭环 系统平均处理时效优于2秒/图 显著提升了多模态图像分析工程的智能化与易用性"}]
    
    for user, bot in history:
        messages.append({"role": "user", "content": user})
        messages.append({"role": "assistant", "content": bot})
    messages.append({"role": "user", "content": message})

    try:
        response = requests.post(API_URL, headers=headers, json={
            "model": "deepseek-chat",
            "messages": messages,
            "temperature": 0.7
        }, timeout=30)
        return response.json()['choices'][0]['message']['content']
    except Exception as e:
        return f"请求出错：{str(e)}"

# ========== 主界面 ==========
html_content = """
<div style='
    display: flex; 
    justify-content: center; /* 水平居中 */
    align-items: center; /* 垂直居中 */
    height: 100px; /* 设置块的高度 */
    border-radius: 10px; /* 圆角（可选） */
    font-size: 44px; /* 字体大小 */
    text-align: center; /* 文本居中 */
'>
    多模态交互式图像分割系统 2.0
</div>
"""


def create_main_interface():
    with gr.Blocks(title="温浩2021218526", theme=gr.themes.Soft()) as demo:
        with gr.Row():
            gr.Image("assets/logo.png",
                     height=100,
                     show_download_button=False,
                     show_fullscreen_button=False,
                     show_label=False)
        with gr.Row():
            gr.HTML(html_content)
        with gr.Tabs():
            with gr.Tab("📷 图像分割"):
                create_image_segmentation_interface()
            
            # with gr.Tab("🎨 文生图"):
            #     with gr.Row():
            #         with gr.Column(scale=15):
            #             txt1 = gr.Textbox(lines=2, label="")
            #             txt2 = gr.Textbox(lines=2, label="")
            #         with gr.Column(scale=1,min_width=1):
            #             button1 = gr.Button(value="1")
            #             button2 = gr.Button(value="2")
            #             button3 = gr.Button(value="3")
            #             button4 = gr.Button(value="4")
            #         with gr.Column(scale=6):
            #             generate_button = gr.Button(value="Generate",variant="primary",scale=1)
            #             with gr.Row():
            #                 dropdown = gr.Dropdown(["1", "2", "3", "4"], label="Style1")
            #                 dropdown2 = gr.Dropdown(["1", "2", "3", "4"], label="Style2")

            #     with gr.Row():
            #         with gr.Column():
            #             with gr.Row():
            #                 dropdown3 = gr.Dropdown(["1", "2", "3", "4"], label="Sampling method")
            #                 slider1 = gr.Slider(minimum=0, maximum=1, label="Sampling steps")
            #             checkboxgroup = gr.CheckboxGroup(["Restore faces", "Tiling", "Hires.fix"], label="")
            #             with gr.Row():
            #                 slider2 = gr.Slider(minimum=0, maximum=100, label="Width")
            #                 slider3 = gr.Slider(minimum=0, maximum=100, label="Batch count")
            #             with gr.Row():
            #                 slider4 = gr.Slider(minimum=0, maximum=100, label="Height")
            #                 slider5 = gr.Slider(minimum=0, maximum=100, label="Batch size")
            #             slider6 = gr.Slider(minimum=0, maximum=100, label="CFG scale")
            #             with gr.Row():
            #                 number = gr.Number(label="Seed")
            #                 button5 = gr.Button(value="Randomize")
            #                 button6 = gr.Button(value="Reset")
            #                 checkbox1 = gr.Checkbox(label="Extra")
            #             dropdown4 = gr.Dropdown(["1", "2", "3", "4"], label="Script")
            #         with gr.Column():
            #             gallery = gr.Gallery([],columns=3)
            #             with gr.Row():
            #                 button66 = gr.Button(value="Save")
            #                 button7 = gr.Button(value="Save as")
            #                 button8 = gr.Button(value="Zip")
            #                 button9 = gr.Button(value="Send to img2img",min_width=1)
            #                 button10 = gr.Button(value="Send to inpaint",min_width=1)
            #                 button11 = gr.Button(value="Send to extras",min_width=1)
            #             txt3 = gr.Textbox(lines=4, label="")

            with gr.Tab("💬 AI助手"):
                gr.ChatInterface(
                    fn=doChatbot,
                    chatbot=gr.Chatbot(height=500, bubble_full_width=False),
                    textbox=gr.Textbox(placeholder="输入关于图像处理的问题...", container=False),
                    title="🤖 AI助手",
                    examples=[
                        ["如何提高分割精度？"],
                        ["如何使用这个系统？"],
                        ["解释下匹配阈值的作用"]
                    ],
                    submit_btn="发送",
                    retry_btn="重试",
                    undo_btn="撤回",
                    clear_btn="清空"
                )

    return demo

# ========== 运行程序 ==========
if __name__ == "__main__":
    app = create_main_interface()
    app.launch(
        server_port=8000,
        share=True,
        # favicon="assets/favicon.ico",
        auth=("wh2021218526", "hfut123456")  # 账号and密码
    )