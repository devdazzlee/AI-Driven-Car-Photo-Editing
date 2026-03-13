"""
AI-Driven Car Photo Enhancement Pipeline (enhance-preserve mode).

Uses FLUX Fill Pro for reflection removal (body, glass, ceiling).
Uses OpenCV for tires (avoids NSFW filter).
Color correction applied after FLUX to match surrounding pixels.
"""

import io
from typing import Optional

import cv2
import numpy as np
from PIL import Image as PILImage

from app.services.image_utils import load_image
from app.services.replicate_service import flux_fill_pro_inpaint


class EnhancePreserveService:
    """Main pipeline service for enhance-preserve processing mode."""

    _instance: Optional["EnhancePreserveService"] = None

    def __new__(cls, rmbg_pipeline=None) -> "EnhancePreserveService":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._rmbg_pipeline = rmbg_pipeline
        return cls._instance

    def _get_pipeline(self):
        """Get RMBG pipeline (lazy load from background_removal if not set)."""
        if self._rmbg_pipeline is None:
            from app.services.background_removal import background_removal_service

            self._rmbg_pipeline = background_removal_service._get_pipeline()
        return self._rmbg_pipeline

    # =====================================================================
    # STEP 1: CAR MASK
    # =====================================================================

    def _get_car_mask(self, image: np.ndarray) -> np.ndarray:
        """Generate binary mask of the car using RMBG-1.4."""
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_image = PILImage.fromarray(rgb)

        pipeline = self._get_pipeline()
        pipeline_result = pipeline(pil_image)

        if isinstance(pipeline_result, list) and len(pipeline_result) > 0:
            result_item = pipeline_result[0]
            if isinstance(result_item, dict) and "mask" in result_item:
                mask_pil = result_item["mask"]
            elif hasattr(result_item, "convert"):
                mask_pil = result_item
            else:
                mask_pil = pipeline_result[0]
        elif isinstance(pipeline_result, dict) and "mask" in pipeline_result:
            mask_pil = pipeline_result["mask"]
        elif hasattr(pipeline_result, "convert"):
            mask_pil = pipeline_result
        else:
            mask_pil = pipeline_result

        if hasattr(mask_pil, "mode") and mask_pil.mode == "RGBA":
            mask_gray = np.array(mask_pil.split()[3])
        elif hasattr(mask_pil, "convert"):
            mask_gray = np.array(mask_pil.convert("L"))
        else:
            mask_gray = np.array(mask_pil)
            if len(mask_gray.shape) == 3:
                mask_gray = cv2.cvtColor(mask_gray, cv2.COLOR_RGB2GRAY)

        h, w = image.shape[:2]
        if mask_gray.shape[0] != h or mask_gray.shape[1] != w:
            mask_gray = cv2.resize(mask_gray, (w, h), interpolation=cv2.INTER_NEAREST)

        _, mask_binary = cv2.threshold(mask_gray, 128, 255, cv2.THRESH_BINARY)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        mask_binary = cv2.morphologyEx(mask_binary, cv2.MORPH_CLOSE, kernel)
        mask_binary = cv2.morphologyEx(mask_binary, cv2.MORPH_OPEN, kernel)

        coverage = np.count_nonzero(mask_binary) / mask_binary.size
        print(f"[CAR MASK] coverage={coverage:.3f}")

        return mask_binary

    def _context_inpaint(
        self, image: np.ndarray, mask: np.ndarray, radius: Optional[int] = None
    ) -> np.ndarray:
        """Inpaint using OpenCV - for tires only (avoids NSFW)."""
        if np.count_nonzero(mask) == 0:
            return image
        if radius is None:
            mask_px = np.count_nonzero(mask)
            radius = min(15, max(5, int(np.sqrt(mask_px) / 80)))
        return cv2.inpaint(image, mask, radius, cv2.INPAINT_NS)

    def _color_correct_to_surrounding(
        self, result: np.ndarray, mask: np.ndarray, region_mask: np.ndarray
    ) -> np.ndarray:
        """Adjust inpainted region to match surrounding colors. Fixes FLUX wrong colors."""
        if np.count_nonzero(mask) == 0:
            return result
        surround = (region_mask > 0) & (mask == 0)
        if np.count_nonzero(surround) < 100:
            return result
        src_pixels = result[surround]
        inpaint_pixels = result[mask > 0]
        src_mean = np.mean(src_pixels, axis=0)
        src_std = np.std(src_pixels, axis=0) + 1e-6
        tgt_mean = np.mean(inpaint_pixels, axis=0)
        tgt_std = np.std(inpaint_pixels, axis=0) + 1e-6
        mask_bool = mask > 0
        mask_float = mask_bool.astype(np.float32)
        feather = cv2.GaussianBlur(mask_float, (41, 41), 10)
        if feather.ndim == 2:
            feather = feather[:, :, np.newaxis]
        corrected = result.copy().astype(np.float32)
        for c in range(3):
            ch = corrected[:, :, c]
            ch[mask_bool] = np.clip(
                (ch[mask_bool] - tgt_mean[c]) * (src_std[c] / tgt_std[c]) + src_mean[c],
                0, 255,
            )
            corrected[:, :, c] = ch
        return (
            corrected.astype(np.float32) * feather
            + result.astype(np.float32) * (1 - feather)
        ).astype(np.uint8)

    # =====================================================================
    # STEP 2: PROTECTION MASKS
    # =====================================================================

    def _get_protection_masks(self, image: np.ndarray, car_mask: np.ndarray) -> dict:
        """Create masks for floor, wall, and ceiling regions."""
        h, w = image.shape[:2]
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        car_coords = np.where(car_mask > 0)
        if len(car_coords[0]) == 0:
            empty = np.zeros((h, w), dtype=np.uint8)
            return {
                "floor_mask": empty,
                "wall_mask": empty,
                "above_car_mask": empty,
                "combined_protection": empty,
            }

        car_top = int(car_coords[0].min())
        car_bottom = int(car_coords[0].max())

        floor_region_top = int(h * 0.65)
        row_indices = np.arange(h)[:, None] * np.ones((1, w))

        floor_mask = np.zeros((h, w), dtype=np.uint8)
        floor_by_color = (
            (gray < 130) & (car_mask == 0) & (row_indices >= floor_region_top)
        )
        floor_mask[floor_by_color] = 255

        below_car = np.zeros((h, w), dtype=np.uint8)
        below_car[car_bottom:, :] = 255
        below_car[car_mask > 0] = 0
        floor_mask = cv2.bitwise_or(floor_mask, below_car)

        floor_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 5))
        floor_mask = cv2.morphologyEx(floor_mask, cv2.MORPH_CLOSE, floor_kernel)

        above_car_mask = np.zeros((h, w), dtype=np.uint8)
        above_car_mask[:car_top, :] = 255

        wall_mask = np.zeros((h, w), dtype=np.uint8)
        wall_mask[:, :] = 255
        wall_mask[car_mask > 0] = 0
        wall_mask[floor_mask > 0] = 0
        wall_mask[above_car_mask > 0] = 0

        combined_protection = cv2.bitwise_or(floor_mask, wall_mask)

        print(
            f"[MASKS] floor={np.count_nonzero(floor_mask)} px, "
            f"wall={np.count_nonzero(wall_mask)} px, "
            f"above_car={np.count_nonzero(above_car_mask)} px"
        )

        return {
            "floor_mask": floor_mask,
            "wall_mask": wall_mask,
            "above_car_mask": above_car_mask,
            "combined_protection": combined_protection,
        }

    # =====================================================================
    # STEP 3: CEILING LIGHT REMOVAL (FLUX Fill Pro)
    # =====================================================================

    def _remove_ceiling_lights(
        self,
        image: np.ndarray,
        above_car_mask: np.ndarray,
        car_mask: np.ndarray,
    ) -> np.ndarray:
        """Remove ceiling lights, pipes, fixtures using FLUX Fill Pro.
        Only detects discrete dark objects (not broad gray areas) to avoid huge masks that timeout."""
        print("[CEILING] Starting ceiling light removal...")

        if np.count_nonzero(above_car_mask) == 0:
            print("[CEILING] No area above car found")
            return image

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Only detect clearly dark objects (lights, pipes) — gray < 120
        # Avoid gray < 200 which catches too much and creates 1M+ px masks that timeout
        dark_raw = np.zeros(image.shape[:2], dtype=np.uint8)
        dark_raw[((gray < 120) & (above_car_mask > 0) & (car_mask == 0))] = 255

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        dark_raw = cv2.morphologyEx(dark_raw, cv2.MORPH_CLOSE, kernel)
        dark_raw = cv2.morphologyEx(dark_raw, cv2.MORPH_OPEN, kernel)

        # Keep only discrete objects (50–300K px each) — filter out noise and huge blobs
        contours, _ = cv2.findContours(
            dark_raw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        ceiling_objects = np.zeros(image.shape[:2], dtype=np.uint8)
        for c in contours:
            area = cv2.contourArea(c)
            if 50 < area < 300000:
                cv2.drawContours(ceiling_objects, [c], -1, 255, -1)

        ceiling_objects = cv2.dilate(ceiling_objects, kernel, iterations=2)
        ceiling_objects = cv2.bitwise_and(ceiling_objects, above_car_mask)

        detected_px = int(np.count_nonzero(ceiling_objects))
        print(f"[CEILING] Detected {detected_px} px of ceiling objects")

        if detected_px < 500:
            print("[CEILING] Nothing significant to remove")
            return image

        if detected_px > 400000:
            result = self._context_inpaint(image, ceiling_objects, radius=9)
            fill = np.full_like(image, [255, 255, 255], dtype=np.uint8)
            mask_float = (ceiling_objects / 255.0).astype(np.float32)[:, :, np.newaxis]
            result = (
                result.astype(np.float32) * (1 - mask_float)
                + fill.astype(np.float32) * mask_float
            ).astype(np.uint8)
            print(f"[CEILING] Removed {detected_px} px via context inpainting (large mask)")
        else:
            result = flux_fill_pro_inpaint(
                image=image,
                mask=ceiling_objects,
                prompt="Clean smooth white studio wall and ceiling, plain white, no lights no pipes",
                guidance=30,
                steps=35,
            )
            print(f"[CEILING] Removed {detected_px} px via FLUX Fill Pro")
        return result

    # =====================================================================
    # STEP 4: BODY REFLECTION REMOVAL (FLUX Fill Pro)
    # =====================================================================

    def _remove_body_reflections(
        self, image: np.ndarray, car_mask: np.ndarray
    ) -> np.ndarray:
        """Remove specular reflections from car body paint using context inpainting.
        Samples colors from surrounding pixels - preserves exact car color, logos."""
        print("[BODY REFL] Starting body reflection detection...")

        h, w = image.shape[:2]
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l_channel = lab[:, :, 0].astype(np.float32)
        l_local_avg = cv2.GaussianBlur(l_channel, (0, 0), sigmaX=40)
        brightness_excess = l_channel - l_local_avg

        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        exclude_bright_features = (
            (hsv[:, :, 1] > 100)
            | ((hsv[:, :, 2] > 210) & (hsv[:, :, 1] > 25))
            | (hsv[:, :, 2] > 240)
        )

        exclude_chrome_logo = (
            (hsv[:, :, 1] < 45) & (hsv[:, :, 2] > 165) & (car_mask > 0)
        )

        reflection_mask = np.zeros(image.shape[:2], dtype=np.uint8)
        is_reflection = (
            (brightness_excess > 25)
            & (car_mask > 0)
            & (~exclude_bright_features)
            & (~exclude_chrome_logo)
        )
        reflection_mask[is_reflection] = 255

        contours, _ = cv2.findContours(
            reflection_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        for c in contours:
            area = cv2.contourArea(c)
            if 400 < area < 18000:
                perimeter = cv2.arcLength(c, True)
                if perimeter > 0 and (perimeter * perimeter) / (area + 1e-6) < 25:
                    cv2.drawContours(reflection_mask, [c], -1, 0, -1)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        reflection_mask = cv2.morphologyEx(reflection_mask, cv2.MORPH_CLOSE, kernel)
        reflection_mask = cv2.morphologyEx(reflection_mask, cv2.MORPH_OPEN, kernel)
        reflection_mask = cv2.dilate(reflection_mask, kernel, iterations=2)
        reflection_mask = cv2.bitwise_and(reflection_mask, car_mask)

        detected_px = int(np.count_nonzero(reflection_mask))
        print(f"[BODY REFL] Detected {detected_px} px of specular highlights")

        if detected_px < 500:
            print("[BODY REFL] No significant reflections found")
            return image

        result = flux_fill_pro_inpaint(
            image=image,
            mask=reflection_mask,
            prompt="Smooth car body paint, same color as surrounding area, no reflections no highlights, matte automotive paint matching this car exactly",
            guidance=28,
            steps=40,
        )
        result = self._color_correct_to_surrounding(result, reflection_mask, car_mask)

        print(f"[BODY REFL] Removed {detected_px} px via FLUX + color correction")
        return result

    # =====================================================================
    # STEP 5: GLASS/WINDSHIELD REFLECTION REMOVAL (FLUX Fill Pro)
    # =====================================================================

    def _remove_glass_reflections(
        self, image: np.ndarray, car_mask: np.ndarray
    ) -> np.ndarray:
        """Remove reflections from car windows using FLUX Fill Pro."""
        h, w = image.shape[:2]
        print("[GLASS] Starting glass reflection detection...")

        car_coords = np.where(car_mask > 0)
        if len(car_coords[0]) == 0:
            print("[GLASS] No car found")
            return image

        car_top = int(car_coords[0].min())
        car_bottom = int(car_coords[0].max())
        car_height = car_bottom - car_top

        window_region_bottom = car_top + int(car_height * 0.72)

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        window_region = np.zeros((h, w), dtype=np.uint8)
        window_region[car_top:window_region_bottom, :] = 255

        dark_glass = np.zeros((h, w), dtype=np.uint8)
        dark_glass[(gray < 140) & (car_mask > 0) & (window_region > 0)] = 255

        kernel_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (55, 55))
        glass_mask = cv2.dilate(dark_glass, kernel_dilate)

        if np.count_nonzero(dark_glass) < 3000:
            car_left = int(car_coords[1].min())
            car_right = int(car_coords[1].max())
            car_width = car_right - car_left
            center_left = car_left + int(car_width * 0.2)
            center_right = car_right - int(car_width * 0.2)
            center_band = np.zeros((h, w), dtype=np.uint8)
            center_band[:, center_left:center_right] = 255
            bright_low_sat = (
                (gray > 110)
                & (gray < 230)
                & (hsv[:, :, 1] < 55)
                & (car_mask > 0)
                & (window_region > 0)
                & (center_band > 0)
            )
            glass_mask = cv2.bitwise_or(
                glass_mask,
                (bright_low_sat.astype(np.uint8) * 255),
            )

        glass_mask = cv2.bitwise_and(glass_mask, car_mask)
        glass_mask = cv2.bitwise_and(glass_mask, window_region)

        glass_px = int(np.count_nonzero(glass_mask))
        print(f"[GLASS] Glass area detected: {glass_px} px")

        if glass_px < 1000:
            print("[GLASS] No significant glass area found")
            return image

        gray_float = gray.astype(np.float32)
        local_mean = cv2.GaussianBlur(gray_float, (41, 41), 12)
        brightness_diff = gray_float - local_mean

        reflection_mask = np.zeros((h, w), dtype=np.uint8)
        method1 = (brightness_diff > 18) & (gray > 70) & (glass_mask > 0)
        reflection_mask[method1] = 255
        method2 = (gray > 130) & (glass_mask > 0)
        reflection_mask[method2] = 255
        method3 = (hsv[:, :, 1] < 50) & (hsv[:, :, 2] > 100) & (glass_mask > 0)
        reflection_mask[method3] = 255
        method4 = (gray > 80) & (local_mean < 70) & (glass_mask > 0)
        reflection_mask[method4] = 255

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        reflection_mask = cv2.morphologyEx(reflection_mask, cv2.MORPH_CLOSE, kernel)
        reflection_mask = cv2.morphologyEx(reflection_mask, cv2.MORPH_OPEN, kernel)
        reflection_mask = cv2.dilate(reflection_mask, kernel, iterations=2)
        reflection_mask = cv2.bitwise_and(reflection_mask, glass_mask)

        reflection_px = int(np.count_nonzero(reflection_mask))
        print(f"[GLASS] Reflection mask: {reflection_px} px")

        if reflection_px < 100:
            print("[GLASS] No reflections found on glass")
            return image

        result = flux_fill_pro_inpaint(
            image=image,
            mask=reflection_mask,
            prompt="Car window glass, dark tinted, no reflections no glare, smooth uniform glass matching surrounding windows",
            guidance=30,
            steps=45,
        )
        result = self._color_correct_to_surrounding(result, reflection_mask, glass_mask)

        print(f"[GLASS] Removed {reflection_px} px via FLUX + color correction")
        return result

    # =====================================================================
    # STEP 6: DUST REMOVAL (OpenCV)
    # =====================================================================

    def _clean_dust(
        self,
        image: np.ndarray,
        car_mask: np.ndarray,
        floor_mask: np.ndarray,
    ) -> np.ndarray:
        """Remove small dust particles from car body using OpenCV."""
        print("[DUST] Starting dust detection...")

        work_mask = car_mask.copy()
        work_mask[floor_mask > 0] = 0

        if np.count_nonzero(work_mask) == 0:
            return image

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        combined_dust = np.zeros(image.shape[:2], dtype=np.uint8)
        for ksize in [3, 5, 7, 9]:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
            tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel)
            bright_dust = ((tophat > 10) & (work_mask > 0)).astype(np.uint8) * 255
            combined_dust = cv2.bitwise_or(combined_dust, bright_dust)
        for ksize in [3, 5, 7]:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
            blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
            dark_dust = ((blackhat > 12) & (work_mask > 0)).astype(np.uint8) * 255
            combined_dust = cv2.bitwise_or(combined_dust, dark_dust)

        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(combined_dust)
        for i in range(1, num_labels):
            if stats[i, cv2.CC_STAT_AREA] > 300:
                combined_dust[labels == i] = 0

        dust_px = int(np.count_nonzero(combined_dust))
        print(f"[DUST] Detected {dust_px} px of dust")

        if dust_px < 10:
            return image

        result = cv2.inpaint(image, combined_dust, 2, cv2.INPAINT_TELEA)
        print(f"[DUST] Removed {dust_px} px")
        return result

    # =====================================================================
    # STEP 7: TIRE CLEANUP (FLUX Fill Pro)
    # =====================================================================

    def _clean_tires(self, image: np.ndarray, car_mask: np.ndarray) -> np.ndarray:
        """Clean and blacken tire rubber using FLUX Fill Pro."""
        h, w = image.shape[:2]
        print("[TIRE] Starting tire detection...")

        car_coords = np.where(car_mask > 0)
        if len(car_coords[0]) == 0:
            return image

        car_top = int(car_coords[0].min())
        car_bottom = int(car_coords[0].max())
        car_left = int(car_coords[1].min())
        car_right = int(car_coords[1].max())
        car_height = car_bottom - car_top
        car_width = car_right - car_left

        tire_region_top = car_bottom - int(car_height * 0.28)

        zones = [
            (tire_region_top, car_bottom, car_left, car_left + int(car_width * 0.30)),
            (tire_region_top, car_bottom, car_right - int(car_width * 0.30), car_right),
        ]

        result = image.copy()
        total_tire_px = 0

        for zone_idx, (zt, zb, zl, zr) in enumerate(zones):
            zt = max(0, zt)
            zb = min(h, zb)
            zl = max(0, zl)
            zr = min(w, zr)

            if zt >= zb or zl >= zr:
                continue

            zone_img = image[zt:zb, zl:zr]
            zone_car_mask = car_mask[zt:zb, zl:zr]
            zone_gray = cv2.cvtColor(zone_img, cv2.COLOR_BGR2GRAY)
            zone_hsv = cv2.cvtColor(zone_img, cv2.COLOR_BGR2HSV)
            zone_h = zb - zt

            tire_mask_zone = np.zeros(zone_img.shape[:2], dtype=np.uint8)
            tire_condition = (
                (zone_hsv[:, :, 2] < 95)
                & (zone_hsv[:, :, 1] < 100)
                & (zone_gray < 100)
                & (zone_car_mask > 0)
            )
            tire_mask_zone[tire_condition] = 255

            rim_condition = (zone_gray > 110) | (zone_hsv[:, :, 2] > 120)
            tire_mask_zone[rim_condition] = 0

            arch_cutoff = int(zone_h * 0.35)
            tire_mask_zone[:arch_cutoff, :] = 0

            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
            tire_mask_zone = cv2.morphologyEx(tire_mask_zone, cv2.MORPH_OPEN, kernel)
            tire_mask_zone = cv2.morphologyEx(tire_mask_zone, cv2.MORPH_CLOSE, kernel)

            num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(tire_mask_zone)
            filtered = np.zeros_like(tire_mask_zone)
            for i in range(1, num_labels):
                area = stats[i, cv2.CC_STAT_AREA]
                if area > 2000:
                    filtered[labels == i] = 255
            tire_mask_zone = filtered

            tire_px = int(np.count_nonzero(tire_mask_zone))
            total_tire_px += tire_px
            print(f"[TIRE] Zone {zone_idx}: {tire_px} px detected")

            if tire_px < 2000:
                continue

            full_tire_mask = np.zeros((h, w), dtype=np.uint8)
            full_tire_mask[zt:zb, zl:zr] = tire_mask_zone

            result = self._context_inpaint(result, full_tire_mask, radius=5)

        print(f"[TIRE] Total: {total_tire_px} px processed via context inpainting")
        return result

    # =====================================================================
    # STEP 8: CAR ENHANCEMENT (OpenCV)
    # =====================================================================

    def _enhance_car(self, image: np.ndarray, car_mask: np.ndarray) -> np.ndarray:
        """Subtle sharpening and contrast improvement on car region."""
        print("[ENHANCE] Applying car enhancement...")

        result = image.copy()
        car_px = car_mask > 0

        blurred = cv2.GaussianBlur(image, (0, 0), sigmaX=2)
        sharpened = cv2.addWeighted(image, 1.08, blurred, -0.08, 0)
        result[car_px] = sharpened[car_px]

        lab = cv2.cvtColor(result, cv2.COLOR_BGR2LAB)
        clahe = cv2.createCLAHE(clipLimit=1.2, tileGridSize=(8, 8))
        l_enhanced = clahe.apply(lab[:, :, 0])
        lab[:, :, 0][car_px] = l_enhanced[car_px]
        result = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

        hsv = cv2.cvtColor(result, cv2.COLOR_BGR2HSV)
        s_channel = hsv[:, :, 1].astype(np.float32)
        s_channel[car_px] = np.clip(s_channel[car_px] * 1.02, 0, 255)
        hsv[:, :, 1] = s_channel.astype(np.uint8)
        result = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

        print("[ENHANCE] Applied sharpening + CLAHE + saturation boost")
        return result

    # =====================================================================
    # STEP 9: LIGHTING ADJUSTMENT (OpenCV)
    # =====================================================================

    def _apply_lighting(self, image: np.ndarray, boost: float) -> np.ndarray:
        """Adjust global image brightness."""
        if abs(boost - 1.0) < 0.01:
            print("[LIGHTING] No adjustment needed (boost=1.0)")
            return image

        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l_channel = lab[:, :, 0].astype(np.float32)
        l_channel = np.clip(l_channel * boost, 0, 255).astype(np.uint8)
        lab[:, :, 0] = l_channel
        result = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

        print(f"[LIGHTING] Applied brightness boost={boost}")
        return result

    # =====================================================================
    # STEP 10: FINAL COMPOSITE
    # =====================================================================

    def _final_composite(
        self,
        processed: np.ndarray,
        original: np.ndarray,
        floor_mask: np.ndarray,
        wall_mask: np.ndarray,
        car_mask: np.ndarray,
    ) -> np.ndarray:
        """Restore floor and wall from original image."""
        print("[COMPOSITE] Starting final composite...")

        result = processed.copy()
        protection = cv2.bitwise_or(floor_mask, wall_mask)
        result[protection > 0] = original[protection > 0]

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        dilated_protection = cv2.dilate(protection, kernel, iterations=1)
        eroded_protection = cv2.erode(protection, kernel, iterations=1)
        boundary = cv2.subtract(dilated_protection, eroded_protection)
        boundary_float = cv2.GaussianBlur(boundary, (9, 9), 3).astype(np.float32) / 255.0

        for c in range(3):
            blended = (
                original[:, :, c].astype(np.float32) * boundary_float
                + processed[:, :, c].astype(np.float32) * (1.0 - boundary_float)
            ).astype(np.uint8)
            result[:, :, c] = np.where(boundary > 0, blended, result[:, :, c])

        car_coords = np.where(car_mask > 0)
        if len(car_coords[0]) > 0:
            car_top = int(car_coords[0].min())
            above_car_protected = int(np.count_nonzero(protection[:car_top, :]))
            print(f"[COMPOSITE] Protected above car top: {above_car_protected} (should be 0)")
        total_protected = int(np.count_nonzero(protection))
        print(f"[COMPOSITE] Total protected: {total_protected} px")

        return result

    # =====================================================================
    # MAIN PIPELINE
    # =====================================================================

    def process(
        self,
        image_data: bytes,
        filename: str,
        output_format: str = "png",
        remove_sky_ceiling: bool = True,
        enhance_car: bool = True,
        lighting_boost: float = 1.0,
        car_sharpness: float = 1.08,
        car_contrast: float = 1.0,
    ) -> bytes:
        """
        Main processing pipeline for enhance-preserve mode.
        Compatible with existing processor.py interface.
        """
        pil_img = load_image(image_data, filename)
        image = np.array(pil_img)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        original = image.copy()

        h, w = image.shape[:2]
        print(f"\n{'='*60}")
        print(f"[ENHANCE] START {filename} size=({w}, {h}) lighting={lighting_boost}")
        print(f"{'='*60}")

        print("\n--- Step 1: Car Mask ---")
        car_mask = self._get_car_mask(image)

        print("\n--- Step 2: Protection Masks ---")
        masks = self._get_protection_masks(image, car_mask)

        if remove_sky_ceiling:
            print("\n--- Step 3: Ceiling Lights ---")
            try:
                image = self._remove_ceiling_lights(
                    image, masks["above_car_mask"], car_mask
                )
            except Exception as e:
                print(f"[CEILING] FAILED: {e}")
                import traceback
                traceback.print_exc()

        print("\n--- Step 4: Body Reflections ---")
        try:
            image = self._remove_body_reflections(image, car_mask)
        except Exception as e:
            print(f"[BODY REFL] FAILED: {e}")
            import traceback
            traceback.print_exc()

        print("\n--- Step 5: Glass Reflections ---")
        try:
            image = self._remove_glass_reflections(image, car_mask)
        except Exception as e:
            print(f"[GLASS] FAILED: {e}")
            import traceback
            traceback.print_exc()

        print("\n--- Step 6: Dust Removal (OpenCV) ---")
        try:
            image = self._clean_dust(image, car_mask, masks["floor_mask"])
        except Exception as e:
            print(f"[DUST] FAILED: {e}")
            import traceback
            traceback.print_exc()

        print("\n--- Step 7: Tire Cleanup ---")
        try:
            image = self._clean_tires(image, car_mask)
        except Exception as e:
            print(f"[TIRE] FAILED: {e}")
            import traceback
            traceback.print_exc()

        if enhance_car:
            print("\n--- Step 8: Car Enhancement ---")
            try:
                image = self._enhance_car(image, car_mask)
            except Exception as e:
                print(f"[ENHANCE] FAILED: {e}")
                import traceback
                traceback.print_exc()

        print("\n--- Step 9: Lighting Adjustment ---")
        try:
            image = self._apply_lighting(image, lighting_boost)
        except Exception as e:
            print(f"[LIGHTING] FAILED: {e}")
            import traceback
            traceback.print_exc()

        print("\n--- Step 10: Final Composite ---")
        image = self._final_composite(
            image, original, masks["floor_mask"], masks["wall_mask"], car_mask
        )

        print(f"\n{'='*60}")
        print(f"[ENHANCE] DONE {filename}")
        print(f"{'='*60}\n")

        if output_format.lower() == "png":
            success, buffer = cv2.imencode(".png", image)
        elif output_format.lower() == "webp":
            success, buffer = cv2.imencode(
                ".webp", image, [cv2.IMWRITE_WEBP_QUALITY, 95]
            )
        else:
            success, buffer = cv2.imencode(
                ".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 95]
            )

        if not success:
            raise RuntimeError(f"Failed to encode image as {output_format}")

        return buffer.tobytes()


enhance_preserve_service = EnhancePreserveService()
