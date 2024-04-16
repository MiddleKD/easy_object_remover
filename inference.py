from kandinsky_inpaint_adapter import KandinskyInpaintAdapter
from simple_lama_inpainting import SimpleLama
from PIL import Image

def load_lama_remover(model_path, device):
    remover = SimpleLama(model_path, device)
    return remover

def load_kandinsky_remover(model_path, 
                           image_processor_path, 
                           image_encoder_path,
                           device):
    remover = KandinskyInpaintAdapter(
        model_path=model_path,
        image_processor_path=image_processor_path,
        image_encoder_path=image_encoder_path,
        device=device
    )
    return remover

def inferene(image, mask_image, model):
    image = image.convert("RGB")
    mask_image = mask_image.convert("L")

    result_image = model(image, mask_image)
    return result_image

if __name__ == "__main__":
    remover = load_lama_remover(
        model_path="/home/mlfavorfit/Desktop/lib_link/favorfit/kjg/0_model_weights/lama_inpainting/big-lama.pt",
        device="cuda"
    )
    
    remover2 = load_kandinsky_remover(
        model_path="/home/mlfavorfit/Desktop/lib_link/favorfit/kjg/0_model_weights/diffusion/Kandinsky/kandinsky-decoder-inpainting",
        image_processor_path="/home/mlfavorfit/Desktop/lib_link/favorfit/kjg/0_model_weights/diffusion/Kandinsky/kandinsky-prior/image_processor",
        image_encoder_path="/home/mlfavorfit/Desktop/lib_link/favorfit/kjg/0_model_weights/diffusion/Kandinsky/kandinsky-prior/image_encoder",
        device="cuda"
    )

    image = Image.open("/home/mlfavorfit/Downloads/text_sample/pr_watch.jpg")
    mask_image = Image.open("/home/mlfavorfit/Downloads/text_sample/mask_watch.png")
    
    result_image = inferene(image, mask_image, remover)
    result_image.show()

    result_image = inferene(image, mask_image, remover2)
    result_image.show()