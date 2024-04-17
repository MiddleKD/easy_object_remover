import os
from PIL import Image
import torch
import numpy as np

from diffusers import KandinskyV22InpaintPipeline
from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection

from kandinsky_inpaint_adapter.utils import (make_positive_style, 
                                             make_negative_style,
                                             prepare_img_and_mask)


class KandinskyInpaintAdapter:
    def __init__(self, 
                 model_path, 
                 image_processor_path, 
                 image_encoder_path, 
                 device):

        self.device = device
        self.image_processor = CLIPImageProcessor.from_pretrained(image_processor_path)
        self.image_encoder = CLIPVisionModelWithProjection.from_pretrained(image_encoder_path)
        self.pipeline = KandinskyV22InpaintPipeline.from_pretrained(
            model_path, 
            torch_dtype=torch.float16, 
            use_safetensors=True
        ).to(device)
    
    def get_image_embedding(self, image: Image.Image):
        clip_images = self.image_processor(image, return_tensors="pt").pixel_values
        image_embeds = self.image_encoder(clip_images).image_embeds.to(dtype=torch.float16).to(self.device)
        return image_embeds

    def __call__(self, image: Image.Image, mask: Image.Image, seed=-1, use_negative_embedding_file=False):
        image, mask = prepare_img_and_mask(image, mask)
        width, height = image.size

        positive_style = make_positive_style(image, mask)
        positive_embeds = self.get_image_embedding(positive_style)
        
        if use_negative_embedding_file == True:
            negative_embeds = torch.load(os.path.join(os.path.dirname(__file__), "negative_embedding.pth"))
        else:
            negative_style = make_negative_style(image, mask)
            negative_embeds = self.get_image_embedding(negative_style)

        generator = torch.Generator()
        if seed == -1:
            seed = np.random.randint(0, 10**9)
        generator.manual_seed(seed)

        output = self.pipeline(image=image,
            mask_image=mask, 
            image_embeds=positive_embeds,
            negative_image_embeds=negative_embeds,
            height=height,
            width=width,
            guidance_scale=0.0,
            num_inference_steps=20,
            strength=1.0,
            generator=generator
        ).images[0]
        
        return output
    