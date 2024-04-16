import time
from PIL import Image
import numpy as np
import gradio as gr

from inference import load_lama_remover, load_kandinsky_remover, inferene
from Favorfit_OCR.inference import call_easyocr
from Favorfit_OCR.inference import inference as ocr_inference

lama_remover = load_lama_remover(
    model_path="/home/mlfavorfit/Desktop/lib_link/favorfit/kjg/0_model_weights/lama_inpainting/big-lama.pt",
    device="cuda"
)
    
kandinsky_remover = load_kandinsky_remover(
    model_path="/home/mlfavorfit/Desktop/lib_link/favorfit/kjg/0_model_weights/diffusion/Kandinsky/kandinsky-decoder-inpainting",
    image_processor_path="/home/mlfavorfit/Desktop/lib_link/favorfit/kjg/0_model_weights/diffusion/Kandinsky/kandinsky-prior/image_processor",
    image_encoder_path="/home/mlfavorfit/Desktop/lib_link/favorfit/kjg/0_model_weights/diffusion/Kandinsky/kandinsky-prior/image_encoder",
    device="cuda"
)

ocr_model = call_easyocr(["ko", "en"])

st_width, st_height = (None, None)

def composing_output(img1, img2, mask):
    img1 = np.array(img1)
    mask = np.array(mask)
    img2 = np.array(img2)
    
    composed_output = np.array(img1) * (1-mask/255) + np.array(img2) * (mask/255)
    return Image.fromarray(composed_output.astype(np.uint8))

def resize_store_ratio(image, min_side=512):

    width, height = image.size

    if width < height:
        new_width = min_side
        new_height = int((height / width) * min_side)
    else:
        new_width = int((width / height) * min_side)
        new_height = min_side

    resized_image = image.resize((new_width, new_height))

    return resized_image

def run_inference(edited_image, is_kandinsky, resize_size):
    global st_width, st_height
    image = edited_image["background"]
    mask = edited_image["layers"][0]
    st_width, st_height = image.size

    if all(cur >= resize_size for cur in image.size):
        image = resize_store_ratio(image, resize_size)
        mask = resize_store_ratio(mask, resize_size)

    if is_kandinsky == True:
       output = inferene(image, mask, kandinsky_remover)
    else:
       output = inferene(image, mask, lama_remover)
    
    return output

def run_ocr(image):
    img = image["background"]

    ocr_result, mask_image = ocr_inference(img, ocr_model, conf_threshold=0.5)
    rgba_mask = mask_image.convert("RGB")
    rgba_mask.putalpha(mask_image)
    rgba_mask.convert("RGB").show()
    for cur in ((None, None), ({"background":img, "layers":[rgba_mask], "composite":img}, ocr_result)):
        yield cur
        time.sleep(0.5)

def output_to_input(image):
    image = image.resize((st_width, st_height))
    for cur in (None, image):
        yield cur
        time.sleep(0.5)


js_func = """
function refresh() {
    const url = new URL(window.location);

    if (url.searchParams.get('__theme') !== 'dark') {
        url.searchParams.set('__theme', 'dark');
        window.location.href = url.href;
    }
}
"""

with gr.Blocks(js=js_func) as demo:
    with gr.Row():
        with gr.Column():
            input_image = gr.ImageEditor(
                label="original image", 
                show_label=True, 
                image_mode="RGB",
                type="pil",
                eraser=gr.Eraser(default_size=30),
                brush=gr.Brush(colors=["#FFFFFF"], default_size=30, color_mode="fixed"),
                transforms=(),
                mirror_webcam=False,
                show_download_button=False,
                sources=("upload"),
            )
            ocr_btn = gr.Button(value="Get text mask", variant="secondary")
            ocr_text_box = gr.TextArea(label="Text from image")
        
        with gr.Column():
            output_image = gr.Image(
                label="output image",
                type="pil", 
                show_label=False, 
                image_mode="RGB",
                sources=(),
                interactive=False)
            resize_size_box = gr.Number(value=512, 
                                        precision=0, 
                                        label="resize size",
                                        minimum=256,
                                        maximum=1024, 
                                        info="이미지의 크기가 값보다 크면, 작은 면을 기준으로 크기를 조정합니다.")
            is_kandinsky = gr.Checkbox(value=False, label="assume hidden space")
            run_btn = gr.Button(variant="primary")
            to_input = gr.Button(value="Pass to input", variant="secondary")

    ocr_btn.click(run_ocr, [input_image], [input_image, ocr_text_box])
    run_btn.click(run_inference, [input_image, is_kandinsky, resize_size_box], output_image)
    to_input.click(output_to_input, [output_image], input_image)

demo.launch(share=True)
