from transformers import SegformerImageProcessor, AutoModelForSemanticSegmentation
from PIL import Image
import torch
import torch.nn as nn
import numpy as np

class ClothesSegmenter:
    def __init__(self):
        self.processor = SegformerImageProcessor.from_pretrained("mattmdjaga/segformer_b2_clothes")
        self.model = AutoModelForSemanticSegmentation.from_pretrained("mattmdjaga/segformer_b2_clothes")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        print(f"ClothesSegmenter loaded on {self.device}")

    def segment_image(self, image: Image.Image) -> Image.Image:
        """
        Removes background from the image using SegFormer.
        Returns the image with transparent background (RGBA).
        """
        # Ensure image is RGB
        image = image.convert("RGB")
        
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits

        # Upsample logits to original image size
        upsampled_logits = nn.functional.interpolate(
            logits,
            size=image.size[::-1], # (height, width)
            mode="bilinear",
            align_corners=False,
        )

        pred_seg = upsampled_logits.argmax(dim=1)[0].cpu().numpy()
        
        # Labels: 0 is Background. We want to keep everything else (1-17).
        # Create mask: 0 where label is 0, 255 where label > 0
        mask = np.where(pred_seg == 0, 0, 255).astype(np.uint8)
        
        # Convert mask to PIL Image
        mask_img = Image.fromarray(mask, mode='L')
        
        # Apply mask to alpha channel
        image_rgba = image.convert("RGBA")
        image_rgba.putalpha(mask_img)
        
        return image_rgba

# Singleton instance
segmenter = ClothesSegmenter()
