from .base_cloner import BaseCloner
from openvoice import se_extractor
from openvoice.api import ToneColorConverter
import os
from io import BytesIO
import soundfile as sf
project_root = os.path.join(os.path.dirname(__file__),'..')
ckpt_converter = '/models/checkpoints_v2/converter'
class OpenvoiceCloner(BaseCloner):
    name = "openvoice"
    def __init__(self,device="cpu"):
        self.tone_color_converter = ToneColorConverter(os.path.join(project_root,ckpt_converter,'config.json'), device=device)
        self.tone_color_converter.load_ckpt(os.path.join(project_root,ckpt_converter,'checkpoint.pth'))
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
        #container = BytesIO()
        # if it does not work replace container with "/project/tmp/tmp_file.wav"
        voice_se, voice_name = se_extractor.get_se(voice,self.tone_color_converter,vad=False)
        sample_se, sample_name = se_extractor.get_se(sample,self.tone_color_converter,vad=False)
        cloned_audio = self.tone_color_converter.convert(
            audio_src_path=sample,
            src_se = sample_se,
            tgt_se = voice_se,
            output_path=None # that way it will return audio this way
        )
        #cloned_audio, sr = sf.read(container)
        return cloned_audio, 22050 # taken from here models/checkpoints_v2/converter/config.json