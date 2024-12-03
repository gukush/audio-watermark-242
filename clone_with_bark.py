import os
ROOT_DIR = os.path.dirname(__file__)
import logging
import argparse
import time
import torch, torchaudio
import soundfile as sf
SUPPORTED_AUDIO_EXTENSIONS = ['.wav','.opus','.flac']
import sys
import numpy as np
# This test requires a lot of resources
root_path = os.path.join(os.path.dirname(__file__),'..')#,'bark_with_voice_clone')
sys.path.append(root_path)
submodule_path = os.path.join(root_path,'bark-with-voice-clone')
from bark_with_voice_clone.bark.generation import load_codec_model, generate_text_semantic
from encodec.utils import convert_audio
from bark_with_voice_clone.bark.api import generate_audio, semantic_to_waveform
from transformers import BertTokenizer
from bark_with_voice_clone.bark.generation import SAMPLE_RATE, preload_models, codec_decode, generate_coarse, generate_fine, generate_text_semantic

device = 'cpu'
model = load_codec_model(use_gpu=False)
from bark_with_voice_clone.hubert.hubert_manager import HuBERTManager
os.chdir('/project/')
hubert_manager = HuBERTManager()
hubert_manager.make_sure_hubert_installed()
hubert_manager.make_sure_tokenizer_installed()
from bark_with_voice_clone.hubert.pre_kmeans_hubert import CustomHubert
from bark_with_voice_clone.hubert.customtokenizer import CustomTokenizer
hubert_model = CustomHubert(checkpoint_path='/project/data/models/hubert/hubert.pt',device=device).to(device)
tokenizer = CustomTokenizer.load_from_checkpoint('/project/data/models/hubert/tokenizer.pth',map_location=device).to(device)

def clone_voice_to_sample(sample,voice):
    voice_name, _ = os.path.splitext(os.path.basename(voice))
    prompt_voice_path = os.path.join(submodule_path,'bark','assets','prompts',voice_name+'.npz')
    if os.path.isfile(prompt_voice_path):
        logging.info(f"Extracted voice features for {voice_name} already exist, skipping voice extraction")
    else:
        voice_audio, voice_sr = torchaudio.load(voice)
        voice_wav = convert_audio(voice_audio,voice_sr,model.sample_rate,model.channels).to(device)
        semantic_vectors = hubert_model.forward(voice_wav,input_sample_hz=model.sample_rate)
        semantic_tokens = tokenizer.get_token(semantic_vectors)
        with torch.no_grad():
            encoded_frames = model.encode(voice_wav.unsqueeze(0))
        codes = torch.cat([encoded[0] for encoded in encoded_frames],dim=-1).squeeze()
        codes = codes.cpu().numpy()
        semantic_tokens = semantic_tokens.cpu().numpy()
        np.savez(prompt_voice_path,fine_prompt=codes,coarse_prompt=codes[:2,:],semantic_prompt=semantic_tokens)
    from bark_with_voice_clone.bark.api import generate_audio, semantic_to_waveform
    from transformers import BertTokenizer
    from bark_with_voice_clone.bark.generation import SAMPLE_RATE, preload_models, codec_decode, generate_coarse, generate_fine, generate_text_semantic
    sample_audio, sample_sr = torchaudio.load(sample)
    sample_wav = convert_audio(sample_audio,sample_sr,model.sample_rate, model.channels).to(device)
    semantic_vectors_2 = hubert_model.forward(sample_wav,input_sample_hz=model.sample_rate)
    semantic_tokens_2 = tokenizer.get_token(semantic_vectors_2)
    semantic_tokens_2 = semantic_tokens_2.cpu().numpy()
    cloned_audio = semantic_to_waveform(semantic_tokens_2,history_prompt=voice_name)
    return cloned_audio, model.sample_rate

def voice_clone_samples(samples,voices_list, override=False):
    for voice in voices_list:
        voice_name, _ = os.path.splitext(os.path.basename(voice))
        for sample in samples:
            filename = os.path.basename(sample)
            sample_name, ext = os.path.splitext(filename)
            cloned_path = os.path.join(ROOT_DIR,'audio','clone',f'{sample_name}_bark_{voice_name}{ext}')
            if os.path.isfile(cloned_path) and not override:
                logging.info(f"File {cloned_path} already exists, skipping. Use --override option to change the behavior.")
                continue
            logging.info(f"Processing voice cloning with Bark for sample {filename} with voice {voice_name}")
            start = time.time()
            with torch.no_grad():
                cloned_audio, sr = clone_voice_to_sample(sample,voice)
            end = time.time()
            duration = end - start
            logging.info(f"Ended voice cloning with Bark for sample {filename} with voice {voice_name}, time: {duration}")
            sf.write(cloned_path,cloned_audio,sr)


def main(args):
    os.makedirs(os.path.join(ROOT_DIR,'audio','watermarked'),exist_ok=True)
    os.makedirs(os.path.join(ROOT_DIR, 'audio','clone'),exist_ok=True)
    if args.samples is not None:
        samples = parse_samples(args.samples)
    if args.voices is not None:
        voices = parse_samples(args.voices)
    if args.detect is not None:
        samples_to_detect = parse_samples(args.detect)
    else:
        samples_to_detect = None
    voice_clone_samples(samples,voices,args.override)
    
def parse_technique_list(str,supported_list):
    if str is None:
        return None
    if str == 'all':
        return supported_list[1:] # skip empty string
    used_techniques = []
    names = str.split(',')
    for name in names:
        if '-' in name: # treat it as range of values
            values = name.split('-')
            assert len(values) == 2, "Incorrectly defined range in options"
            assert values[0].isdigit(), "Wrong number for range of values"
            assert values[1].isdigit(), "Wrong number for range of values"
            lower = int(values[0])
            upper = int(values[1])
            used_techniques.extend(supported_list[lower:(upper+1)])
        elif name.isdigit():
            used_techniques.extend([name])
        else:
            # treat it as name of technique
            if name.lower() in supported_list:
                used_techniques.extend([name.lower()])
    used_techniques = list(dict.fromkeys(used_techniques))
    return used_techniques

def parse_samples(samples_list):
    file_list = []
    if samples_list is None:
        return []
    for sample in samples_list:
        if os.path.isdir(sample):
            for item in os.listdir(sample):
                file_path = os.path.join(sample,item)
                if os.path.isfile(file_path):
                    extension = os.path.splitext(file_path)[1]
                    if extension in SUPPORTED_AUDIO_EXTENSIONS:
                        file_list.append(file_path)
                    else:
                        print(f"Warning {file_path} is not a supported audio file")
        elif os.path.isfile(sample):
            extension = os.path.splitext(sample)[1]
            if extension in SUPPORTED_AUDIO_EXTENSIONS:
                file_list.append(sample)
            else:
                print(f"Warning {sample} is not a supported audio file")
    return file_list

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CLI Tool for running audio watermarking and voice cloning pipeline")
    parser.add_argument("--samples",nargs='+',help="Input audio file or directory with audio files that will be used for watermarking")
    parser.add_argument("--watermarks",help="String comma-delimioted of numbers or names or simply \"all\" that indicate watermarking techniques to use in pipeline")
    parser.add_argument("--clones",help="String comma-delimioted of numbers or names or simply \"all\" that indicate voice cloning techniques to use in pipeline",required=False)
    parser.add_argument("--voices",nargs='+',help="Input audio file or directory with audio files that will be used as voice for voice cloning",required=False)
    parser.add_argument("--override",action='store_true')
    parser.add_argument("--detect",nargs='+',help="Inout audio file or directory with audio files that will be object(s) of watermark detection.")
    parser.add_argument("--identity",action='store_true',help="Setting this option to true makes the detect watermark function look fr watermark name in the file and only compare it to that single technique, discarding other possibilites.")
    args = parser.parse_args()
    #parser.add_argument("--distortions",help="String comma-delimioted of numbers or names or simply \"all\" that indicate watermarking techniques to use in pipeline")
    logging.basicConfig(
        level=logging.INFO,
        format = '%(asctime)s - %(levelname)s - %(message)s',
        handlers = [
            logging.FileHandler("log"),
            logging.StreamHandler()
        ]
    )
    main(args)
