import os
import sys
import torch

project_root =  os.path.join(os.path.dirname(__file__),'..')
os.chdir(os.path.join(project_root, 'IMS-Toucan'))
sys.path.append(os.path.join(project_root, 'IMS-Toucan'))

from InferenceInterfaces.ToucanTTSInterface import ToucanTTSInterface

def read_texts(sentence, filename, model_id=None, device="cpu", language="eng", speaker_reference=None, duration_scaling_factor=1.0):
    tts = ToucanTTSInterface(device=device, tts_model_path=model_id)
    tts.set_language(language)
    if speaker_reference is not None:
        tts.set_utterance_embedding(speaker_reference)
    if type(sentence) == str:
        sentence = [sentence]
    tts.read_to_file(text_list=sentence, file_location=filename, duration_scaling_factor=duration_scaling_factor, prosody_creativity=0.0)
    del tts

def polish_test(version, model_id=None, exec_device="cpu", speaker_reference=None):
    os.makedirs(os.path.join(project_root, 'audio', 'toucan'), exist_ok=True)

    read_texts(model_id=model_id,
               sentence=["Rafi nie moja wina że mam pracofobie, na samą myśl o tym że mógłbym Ci pomóc z robotą paraliżuje mnie strach."],
               filename=f"{project_root}/audio/toucan/polish_toucan_test_{version}.wav",
               device=exec_device,
               language="pol",
               speaker_reference=speaker_reference)

if __name__ == '__main__':
    exec_device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"running on {exec_device}")
    os.makedirs(f"{project_root}/audios/speaker_references/", exist_ok=True)
    merged_speaker_references = ["audios/speaker_references/" + ref for ref in
                                     os.listdir("audios/speaker_references/")]
    polish_test(version="version_11", model_id=None, exec_device=exec_device, speaker_reference=merged_speaker_references if merged_speaker_references != [] else None)