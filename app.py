import time
import gradio as gr
from inference import load_lama_remover, load_kandinsky_remover, inferene

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

st_width, st_height = (None, None)

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

def output_to_input(image):
    image = image.resize((st_width, st_height))
    for cur in (None, image):
        yield cur
        time.sleep(1)


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
                eraser=False,
                brush=gr.Brush(colors=["#FFFFFF"], default_size=30, color_mode="fixed"),
                transforms=(),
                mirror_webcam=False,
                show_download_button=False,
                sources=("upload"),
            )
            is_kandinsky = gr.Checkbox(value=False, label="assume hidden space")
            resize_size_box = gr.Number(value=512, 
                                        precision=0, 
                                        label="resize size",
                                        minimum=256,
                                        maximum=1024, 
                                        info="이미지의 크기가 값보다 크면, 작은 면을 기준으로 크기를 조정합니다.")
        
        with gr.Column():
            output_image = gr.Image(
                label="output image",
                type="pil", 
                show_label=False, 
                image_mode="RGB",
                sources=(),
                show_download_button=True)
            btn = gr.Button(variant="primary")
            to_input = gr.Button(value="to input", variant="secondary")

    btn.click(run_inference, [input_image, is_kandinsky, resize_size_box], output_image)
    to_input.click(output_to_input, [output_image], input_image)

demo.launch(share=True)
