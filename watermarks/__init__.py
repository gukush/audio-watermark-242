# Add new watermark techniques here so that its easier to import them
from .simple_watermark import SimpleWatermark
from .base_watermark import BaseWatermark
from .silentcipher_watermark import SilentcipherWatermark
from .audioseal_watermark import AudiosealWatermark
from .wavmark_watermark import WavmarkWatermark

__all__ = [
    "SimpleWatermark",
    "BaseWatermark",
    "AudiosealWatermark",
    "WavmarkWatermark",
    "SilentcipherWatermark"]
