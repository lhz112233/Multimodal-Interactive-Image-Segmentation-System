import gradio as gr
import torch
import clip
from segment_anything import build_sam, SamAutomaticMaskGenerator
import numpy as np
from PIL import Image, ImageDraw
import os
from datetime import datetime
import pandas as pd
from PIL import Image
from io import BytesIO

# 初始化设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 加载模型
print('开始加载CLIP模型...')
clip_model, preprocess = clip.load("ViT-B/32", device=device)
print('CLIP模型加载完成')

# 加载SAM模型
print('开始加载SAM模型...')
model_path = os.path.join(os.path.dirname(__file__), "path/model3.pth")  # 请确保模型路径正确
if os.path.exists(model_path):
    sam = build_sam(checkpoint=model_path)
    mask_generator = SamAutomaticMaskGenerator(sam)
    print('SAM模型加载完成')
else:
    raise FileNotFoundError(f"Model file not found at {model_path}")


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
        draw.bitmap((0, 0), mask_image, fill=(255, 0, 0, 200))
    return Image.alpha_composite(original_image, overlay_image)


def process_image_batch(images_list, search_text, show_mask, alpha, threshold, box_nms_thresh):
    process_info = {"timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
    mask_generator.box_nms_thresh = box_nms_thresh

    # 分割搜索关键词
    search_terms = [term.strip() for term in search_text.split(',') if term.strip()]
    if not search_terms:
        return [None] * len(images_list), pd.DataFrame(), "未输入有效搜索关键词"

    result_images = []
    all_logs = []
    for idx, image_file in enumerate(images_list, 1):
        try:
            # 直接打开上传的图片文件
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

        except Exception as e:
            all_logs.append(f"图片{idx}处理失败: {str(e)}")
            result_images.append(None)

    log_df = pd.DataFrame({"处理日志": all_logs})
    stats = f"共处理{len(images_list)}张图片 | 总耗时{sum([process_info.get('clip_time', 0)]):.2f}秒"
    return result_images, log_df, stats


def main_interface():
    with gr.Blocks(title="智能图像分割系统") as demo:
        gr.Image("assets/logo.png", scale=1)
        gr.HTML("<div style='text-align: center;'>🖼️ 智能图像分割系统</div>")
        gr.Markdown("上传图片并输入关键词，系统将自动标记匹配区域")

        with gr.Row():  
            input_col = gr.Column()
            output_col = gr.Column()

            with input_col:
                input_images = gr.Files(label="上传图片", file_types=["image"], file_count="multiple")
                search_text = gr.Textbox(label="搜索内容", placeholder="例: 人, 狗, 汽车...", value="Please help me segment a pig.")
                threshold = gr.Slider(0.01, 0.5, value=0.1, label="匹配阈值")
                process_btn = gr.Button("🚀 开始处理", variant="primary")

            with output_col:
                output_gallery = gr.Gallery(label="处理结果", columns=3, height=500)
                logs = gr.Dataframe(label="处理日志", headers=["日志"])
                stats = gr.Textbox(label="统计信息")

        process_btn.click(
            fn=process_image_batch,
            inputs=[input_images, search_text, gr.Checkbox(True, visible=False), gr.Number(0.5, visible=False),
                    threshold, gr.Number(0.7, visible=False)],
            outputs=[output_gallery, logs, stats]
        )

    return demo


if __name__ == "__main__":
    demo = main_interface()
    demo.launch(share=True)