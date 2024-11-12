from .base_cloner import BaseCloner
from TTS.api import TTS
import os
project_root = os.path.join(os.path.dirname(__file__),'..')
class CoquittsCloner(BaseCloner):
    name = "coqui-tts"
    def __init__(self,device="cpu"):
        self.model = TTS(model_name="voice_conversion_models/multilingual/vctk/freevc24", progress_bar=True).to(device)
        super().__init__()
    def extract_voice(self,audio):
        """
        For extracting voice from audio sample to later use for cloning.
        To be implemented by derived class.
        """
        pass
    def apply_voice(self,input, output):
        """
        For transforming voice from one sample to previously extracted one.
        To be implemented by derived class.
        """
        pass
    # returns sampled audio as object
    def clone_voice_to_sample(self, sample, voice):
        """
        Sample and voice are both filepaths.
        """
        cloned_audio = self.model.voice_conversion(source_wav=sample,target_wav=voice)
        return cloned_audio, 24000 # taken from here: https://github.com/OlaWod/FreeVC/blob/main/configs/freevc.json