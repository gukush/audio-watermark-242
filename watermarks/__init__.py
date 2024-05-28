# Add new watermark techniques here so that its easier to import them
from .simple_watermark import SimpleWatermark
from .base_watermark import BaseWatermark
from .wavmark_watermark import WavMarkWatermark

__all__ = ["SimpleWatermark", "BaseWatermark", "WavMarkWatermark"]
