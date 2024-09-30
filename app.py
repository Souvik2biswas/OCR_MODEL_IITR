import gradio as gr
import spaces
import json
from transformers import AutoModel, AutoTokenizer, Qwen2VLForConditionalGeneration, AutoProcessor
import torch
from PIL import Image
import numpy as np
import os
import base64
import io
import uuid
import tempfile
import time
import shutil
from pathlib import Path
import tiktoken
import verovio
model_name = "ucaslcl/GOT-OCR2_0"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
#model = AutoModel.from_pretrained(model_name, trust_remote_code=True, low_cpu_mem_usage=True, device_map='cuda', use_safetensors=True).eval().cuda()
model = AutoModel.from_pretrained(model_name, trust_remote_code=True, low_cpu_mem_usage=True, device_map='cpu', use_safetensors=True).eval()

UPLOAD_FOLDER = "./uploads"
RESULTS_FOLDER = "./results"

for folder in [UPLOAD_FOLDER, RESULTS_FOLDER]:
    if not os.path.exists(folder):
        os.makedirs(folder)

def image_to_base64(image):
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()

q_model_name = "Qwen/Qwen2-VL-2B-Instruct"
#q_model = Qwen2VLForConditionalGeneration.from_pretrained(q_model_name, torch_dtype="auto").cuda().eval()
q_model = Qwen2VLForConditionalGeneration.from_pretrained(q_model_name, torch_dtype="auto").eval()
q_processor = AutoProcessor.from_pretrained(q_model_name, trust_remote_code=True)

def get_qwen_op(image_file, model, processor):
    try:
        image = Image.open(image_file).convert('RGB')
        conversation = [
            {
                "role":"user",
                "content":[
                    {
                        "type":"image",
                    },
                    {
                        "type":"text",
                        "text":"You are an accurate OCR engine. From the given image, extract the Hindi and other text."
                    }
                ]
            }
        ]
        text_prompt = q_processor.apply_chat_template(conversation, add_generation_prompt=True)
        #inputs = q_processor(text=[text_prompt], images=[image], padding=True, return_tensors="pt").to("cuda")
        inputs = q_processor(text=[text_prompt], images=[image], padding=True, return_tensors="pt")
        inputs = {k: v.to(torch.float32) if torch.is_floating_point(v) else v for k, v in inputs.items()}

        generation_config = {
            "max_new_tokens": 1089,
            "do_sample": False,
            "top_k": 20,
            "top_p": 0.90,
            "temperature": 0.4,
            "pad_token_id": q_processor.tokenizer.pad_token_id,
            "eos_token_id": q_processor.tokenizer.eos_token_id,
        }

        output_ids = q_model.generate(**inputs, **generation_config)
        if 'input_ids' in inputs:
                generated_ids = output_ids[:, inputs['input_ids'].shape[1]:]
        else:
            generated_ids = output_ids

        output_text = q_processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)

        return output_text[:] if output_text else "No text extracted from the image."

    except Exception as e:
        return f"An error occurred: {str(e)}"

@spaces.GPU
def run_GOT(image, got_mode, fine_grained_mode="", ocr_color="", ocr_box=""):
    unique_id = str(uuid.uuid4())
    image_path = os.path.join(UPLOAD_FOLDER, f"{unique_id}.png")
    result_path = os.path.join(RESULTS_FOLDER, f"{unique_id}.html")

    shutil.copy(image, image_path)

    try:
        if got_mode == "plain texts OCR":
            res = model.chat(tokenizer, image_path, ocr_type='ocr')
            return res, None
        elif got_mode == "format texts OCR":
            res = model.chat(tokenizer, image_path, ocr_type='format', render=True, save_render_file=result_path)
        elif got_mode == "plain multi-crop OCR":
            res = model.chat_crop(tokenizer, image_path, ocr_type='ocr')
            return res, None
        elif got_mode == "format multi-crop OCR":
            res = model.chat_crop(tokenizer, image_path, ocr_type='format', render=True, save_render_file=result_path)
        elif got_mode == "plain fine-grained OCR":
            res = model.chat(tokenizer, image_path, ocr_type='ocr', ocr_box=ocr_box, ocr_color=ocr_color)
            return res, None
        elif got_mode == "format fine-grained OCR":
            res = model.chat(tokenizer, image_path, ocr_type='format', ocr_box=ocr_box, ocr_color=ocr_color, render=True, save_render_file=result_path)
        elif got_mode == "English + Hindi(Qwen2-VL)":
            res = get_qwen_op(image_path, q_model, q_processor)
            return res, None
        # res_markdown = f"$$ {res} $$"
        res_markdown = res

        if "format" in got_mode and os.path.exists(result_path):
            with open(result_path, 'r') as f:
                html_content = f.read()
            encoded_html = base64.b64encode(html_content.encode('utf-8')).decode('utf-8')
            iframe_src = f"data:text/html;base64,{encoded_html}"
            iframe = f'<iframe src="{iframe_src}" width="100%" height="600px"></iframe>'
            download_link = f'<a href="data:text/html;base64,{encoded_html}" download="result_{unique_id}.html">Download Full Result</a>'
            return res_markdown, f"{download_link}<br>{iframe}"
        else:
            return res_markdown, None
    except Exception as e:
        return f"Error: {str(e)}", None
    finally:
        if os.path.exists(image_path):
            os.remove(image_path)

def task_update(task):
    if "fine-grained" in task:
        return [
            gr.update(visible=True),
            gr.update(visible=False),
            gr.update(visible=False),
        ]
    else:
        return [
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(visible=False),
        ]

def fine_grained_update(task):
    if task == "box":
        return [
            gr.update(visible=False, value = ""),
            gr.update(visible=True),
        ]
    elif task == 'color':
        return [
            gr.update(visible=True),
            gr.update(visible=False, value = ""),
        ]

def search_in_text(text, keywords):
    """Searches for keywords within the text and highlights matches."""

    if not keywords:
        return text

    highlighted_text = text
    for keyword in keywords.split():
        highlighted_text = highlighted_text.replace(keyword, f"<mark>{keyword}</mark>")

    return highlighted_text

def cleanup_old_files():
    current_time = time.time()
    for folder in [UPLOAD_FOLDER, RESULTS_FOLDER]:
        for file_path in Path(folder).glob('*'):
            if current_time - file_path.stat().st_mtime > 3600:  # 1 hour
                file_path.unlink()

title_html = """ OCR Multilingual(GOT OCR 2.O) """

with gr.Blocks() as demo:
    gr.HTML(title_html)
    gr.Markdown("""
    by Souvik Biswas
    ### Guidelines
    Upload your image below and select your preferred mode. Note that more characters may increase wait times.
    - **Plain Texts OCR & Format Texts OCR:** Use these modes for basic image-level OCR.
    - **Plain Multi-Crop OCR & Format Multi-Crop OCR:** Ideal for images with complex content, offering higher-quality results.
    - **Plain Fine-Grained OCR & Format Fine-Grained OCR:** These modes allow you to specify fine-grained regions on the image for more flexible OCR. Regions can be defined by coordinates or colors (red, blue, green, black or white).
    """)

    with gr.Row():
        with gr.Column():
            image_input = gr.Image(type="filepath", label="upload your image")
            task_dropdown = gr.Dropdown(
                choices=[
                    "plain texts OCR",
                    "format texts OCR",
                    "plain multi-crop OCR",
                    "format multi-crop OCR",
                    "plain fine-grained OCR",
                    "format fine-grained OCR",
                    "English + Hindi(Qwen2-VL)"
                ],
                label="Choose one mode of GOT",
                value="plain texts OCR"
            )
            fine_grained_dropdown = gr.Dropdown(
                choices=["box", "color"],
                label="fine-grained type",
                visible=False
            )
            color_dropdown = gr.Dropdown(
                choices=["red", "green", "blue", "black", "white"],
                label="color list",
                visible=False
            )
            box_input = gr.Textbox(
                label="input box: [x1,y1,x2,y2]",
                placeholder="e.g., [0,0,100,100]",
                visible=False
            )
            submit_button = gr.Button("Submit")

        with gr.Column():
            ocr_result = gr.Textbox(label="GOT output")
                # Create the Gradio interface
            iface = gr.Interface(
                fn=search_in_text,
                inputs=[
                    ocr_result,
                    gr.Textbox(label="Keywords",
                        placeholder="search keyword e.g., The",
                        visible=True)],
                outputs=gr.HTML(label="Search Results"),
                allow_flagging="never"
            )
        with gr.Column():
          if ocr_result.value:
            with open("ocr_result.json", "w") as json_file:
              json.dump({"text": ocr_result.value}, json_file) # Access the value of the Textbox using .value

    with gr.Column():
        gr.Markdown("**If you choose the mode with format, the mathpix result will be automatically rendered as follows:**")
        html_result = gr.HTML(label="rendered html", show_label=True)

    task_dropdown.change(
        task_update,
        inputs=[task_dropdown],
        outputs=[fine_grained_dropdown, color_dropdown, box_input]
    )
    fine_grained_dropdown.change(
        fine_grained_update,
        inputs=[fine_grained_dropdown],
        outputs=[color_dropdown, box_input]
    )

    submit_button.click(
        run_GOT,
        inputs=[image_input, task_dropdown, fine_grained_dropdown, color_dropdown, box_input],
        outputs=[ocr_result, html_result]
    )

if __name__ == "__main__":
    cleanup_old_files()
    demo.launch()
