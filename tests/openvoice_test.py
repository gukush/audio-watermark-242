
from openvoice import se_extractor
from openvoice.api import ToneColorConverter
import os

project_root =  os.path.join(os.path.dirname(__file__),'..')
ckpt_converter = '/models/checkpoints_v2/converter'
device = "cpu"
output_dir = 'results'
tone_color_converter = ToneColorConverter(os.path.join(project_root,ckpt_converter,'config.json'), device=device)
tone_color_converter.load_ckpt(os.path.join(project_root,ckpt_converter,'checkpoint.pth'))
reference_speaker = os.path.join(project_root,'audio','old','voice-hispanic-1.wav')
target_se, audio_name = se_extractor.get_se(reference_speaker,tone_color_converter,vad=False)
src_path = os.path.join(project_root,'audio','old','voice-polish-1.wav')
source_se, audio_name_target = se_extractor.get_se(src_path,tone_color_converter,vad=False)
encode_message = "242"
tone_color_converter.convert(audio_src_path = src_path,
    src_se = source_se,
    tgt_se=target_se,
    output_path=os.path.join(project_root,'audio','old','voice-hispanic-1_openvoice.wav'),
    message=encode_message)