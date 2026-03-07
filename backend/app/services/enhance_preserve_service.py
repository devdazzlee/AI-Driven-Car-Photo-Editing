"""Enhance car photos while preserving floor and walls.

- Removes sky/ceiling only (keeps floor & walls)
- Enhances car (sharpening, contrast)
- Adjusts lighting
"""

import io
import logging
from typing import Optional

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

from app.services.image_utils import load_image

logger = logging.getLogger(__name__)

# ADE20K class IDs (SegFormer trained on this)
ADE_SKY = 2
ADE_CEILING = 6
ADE_FLOOR = 5
ADE_WALL = 0

# Classes to remove (replace with inpainting)
REMOVE_CLASSES = [ADE_SKY, ADE_CEILING]


class EnhancePreserveService:
    """Enhance car photos: remove sky/ceiling, keep floor/walls, enhance car, adjust lighting."""

    _instance: Optional["EnhancePreserveService"] = None
    _seg_model = None
    _seg_processor = None

    def __new__(cls) -> "EnhancePreserveService":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def _get_segmentation(self):
        """Lazy-load SegFormer for semantic segmentation."""
        if self._seg_model is None:
            logger.info("Loading SegFormer (ADE20K) for scene segmentation...")
            from transformers import AutoImageProcessor, AutoModelForSemanticSegmentation
            import torch

            self._seg_processor = AutoImageProcessor.from_pretrained(
                "nvidia/segformer-b0-finetuned-ade-512-512"
            )
            self._seg_model = AutoModelForSemanticSegmentation.from_pretrained(
                "nvidia/segformer-b0-finetuned-ade-512-512"
            )
            self._seg_model.eval()
            self._device = "cuda" if __import__("torch").cuda.is_available() else "cpu"
            self._seg_model.to(self._device)
            logger.info("SegFormer loaded successfully")
        return self._seg_processor, self._seg_model

    def _get_remove_mask(self, image: Image.Image) -> np.ndarray:
        """Get binary mask of sky+ceiling to remove."""
        import torch

        processor, model = self._get_segmentation()
        orig_size = image.size  # (W, H)

        # Resize for model
        img_resized = image.resize((512, 512), Image.BILINEAR)
        inputs = processor(images=img_resized, return_tensors="pt")
        inputs = {k: v.to(self._device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)

        # Logits: (1, 150, h, w) - ADE20K has 150 classes
        logits = outputs.logits
        pred = torch.argmax(logits, dim=1).squeeze(0).cpu().numpy()

        # Upsample to original size
        from skimage.transform import resize

        pred_full = resize(
            pred.astype(np.float32),
            (orig_size[1], orig_size[0]),
            order=0,
            preserve_range=True,
            anti_aliasing=False,
        ).astype(np.int64)

        # Mask: True where we want to remove (sky, ceiling)
        mask = np.isin(pred_full, REMOVE_CLASSES)
        return mask.astype(np.uint8)

    def process(
        self,
        image_data: bytes,
        filename: str,
        output_format: str = "png",
        remove_sky_ceiling: bool = True,
        enhance_car: bool = True,
        lighting_boost: float = 1.1,
        car_sharpness: float = 1.3,
        car_contrast: float = 1.1,
    ) -> bytes:
        """
        Process image: remove sky/ceiling, enhance car, adjust lighting.
        """
        img = load_image(image_data, filename)
        if img.mode != "RGB":
            img = img.convert("RGB")

        arr = np.array(img)

        # 1. Remove sky/ceiling (inpainting) - lazy import, can fail on Windows (DLL blocked)
        if remove_sky_ceiling:
            try:
                from skimage.restoration import inpaint_biharmonic

                mask = self._get_remove_mask(img)
                if mask.any():
                    mask_bool = mask.astype(bool)
                    arr = inpaint_biharmonic(arr, mask_bool, channel_axis=2)
            except Exception as e:
                logger.warning("Sky/ceiling removal failed, skipping: %s", e)

        img = Image.fromarray(arr.astype(np.uint8))

        # 2. Car-only enhancement (uses RMBG mask + sharpen/contrast on car region)
        if enhance_car:
            try:
                img = self._enhance_car_region(img, image_data, filename, car_sharpness, car_contrast)
            except Exception as e:
                logger.warning("Car enhancement failed, applying global: %s", e)
                enh = ImageEnhance.Sharpness(img)
                img = enh.enhance(car_sharpness)
                enh = ImageEnhance.Contrast(img)
                img = enh.enhance(car_contrast)

        # 3. Global lighting adjustment
        if lighting_boost != 1.0:
            enh = ImageEnhance.Brightness(img)
            img = enh.enhance(lighting_boost)

        # Save
        buf = io.BytesIO()
        fmt = output_format.upper() if output_format.lower() != "jpg" else "JPEG"
        img.save(buf, format=fmt, quality=95)
        buf.seek(0)
        return buf.read()

    def _enhance_car_region(
        self,
        img: Image.Image,
        original_data: bytes,
        filename: str,
        sharpness: float,
        contrast: float,
    ) -> Image.Image:
        """Enhance only the car region using RMBG mask."""
        from app.services.background_removal import background_removal_service

        # Get car mask from RMBG (foreground = car)
        orig_img = load_image(original_data, filename)
        if orig_img.mode != "RGB":
            orig_img = orig_img.convert("RGB")

        result = background_removal_service._get_pipeline()(orig_img)
        if isinstance(result, Image.Image):
            no_bg = result
        else:
            no_bg = result[0] if isinstance(result, (list, tuple)) else result

        # RMBG returns RGBA - alpha is the mask (255=foreground/car)
        if no_bg.mode == "RGBA":
            car_mask = np.array(no_bg.split()[3])
        else:
            return img

        # Resize mask if needed
        if car_mask.shape[:2] != (img.height, img.width):
            from PIL import Image as PILImage

            mask_pil = PILImage.fromarray(car_mask).convert("L")
            mask_pil = mask_pil.resize((img.width, img.height), PILImage.BILINEAR)
            car_mask = np.array(mask_pil)

        # Create soft mask (0-1) for blending
        car_mask_norm = car_mask.astype(np.float32) / 255.0
        car_mask_3ch = np.stack([car_mask_norm] * 3, axis=-1)

        # Enhanced version (sharpen + contrast)
        img_enhanced = img.filter(ImageFilter.UnsharpMask(radius=1, percent=150, threshold=2))
        enh = ImageEnhance.Contrast(img_enhanced)
        img_enhanced = enh.enhance(contrast)
        enh2 = ImageEnhance.Sharpness(img_enhanced)
        img_enhanced = enh2.enhance(sharpness)

        # Blend: car region gets enhanced, rest stays
        arr_orig = np.array(img).astype(np.float32)
        arr_enh = np.array(img_enhanced).astype(np.float32)
        blended = arr_orig * (1 - car_mask_3ch) + arr_enh * car_mask_3ch
        return Image.fromarray(blended.astype(np.uint8))


enhance_preserve_service = EnhancePreserveService()
