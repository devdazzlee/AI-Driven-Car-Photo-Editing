"""
AI-Driven Car Photo Enhancement Pipeline (enhance-preserve mode).

Uses Gemini API (gemini-3.1-flash-image-preview) for all image processing.
Single API call: remove reflections, clean floor, maintain car color, keep walls/floor intact.
"""

from app.services.gemini_service import process_car_image


class EnhancePreserveService:
    """Main pipeline service for enhance-preserve processing mode."""

    def process(
        self,
        image_data: bytes,
        filename: str,
        output_format: str = "png",
        lighting_boost: float = 1.0,
        **kwargs,
    ) -> bytes:
        """Process car image using Gemini API."""
        return process_car_image(
            image_data,
            filename=filename,
            mode="enhance-preserve",
            output_format=output_format,
            lighting_boost=lighting_boost,
        )


enhance_preserve_service = EnhancePreserveService()
