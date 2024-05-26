# Add new watermark techniques here so that its easier to import them
from .simple_watermark import SimpleWatermark
from .base_watermark import BaseWatermark


__all__ = ["SimpleWatermark", "BaseWatermark"]
