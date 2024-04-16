from PIL import Image
import numpy as np

def make_positive_style(image: Image.Image | np.ndarray, mask: Image.Image | np.ndarray):
    mask_np = np.array(mask.convert("RGB"))
    image_np = np.array(image)

    one_color_image_np = np.ones_like(image_np) * np.mean(image_np, axis=(0,1)).astype(np.uint8)
    positive_style = Image.fromarray(np.where(mask_np > 127, one_color_image_np, image_np))
    
    return positive_style

def make_negative_style(image: Image.Image | np.ndarray, mask: Image.Image | np.ndarray):
    mask_np = np.array(mask.convert("RGB"))
    image_np = np.array(image)

    indices = np.where(mask_np[:,:,0] >= 127)

    y_min, x_min = np.min(indices, axis=1)
    y_max, x_max = np.max(indices, axis=1)

    crop_image = image_np[y_min:y_max, x_min:x_max, :]
    negative_style = Image.fromarray(crop_image)

    return negative_style

def prepare_img_and_mask(image: Image.Image, mask: Image.Image):
    image = image.convert("RGB")
    image = image.resize((round(cur/64) * 64 for cur in image.size))
    
    mask = mask.convert("RGB")
    mask = mask.resize(image.size)

    return image, mask