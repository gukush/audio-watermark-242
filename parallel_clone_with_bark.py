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
root_path = os.path.join(os.path.dirname(__file__))#,'bark_with_voice_clone')
sys.path.append(root_path)
submodule_path = os.path.join(root_path,'bark-with-voice-clone')
from bark_with_voice_clone.bark.generation import load_codec_model, generate_text_semantic
from encodec.utils import convert_audio
from bark_with_voice_clone.bark.api import generate_audio, semantic_to_waveform
from transformers import BertTokenizer
from bark_with_voice_clone.bark.generation import SAMPLE_RATE, preload_models, codec_decode, generate_coarse, generate_fine, generate_text_semantic
#import torch.multiprocessing as mp
import bark_with_voice_clone.bark.generation
from bark_with_voice_clone.bark.api import generate_audio, semantic_to_waveform
from transformers import BertTokenizer
from bark_with_voice_clone.bark.generation import SAMPLE_RATE, preload_models, codec_decode, generate_coarse, generate_fine, generate_text_semantic, _load_model
from bark_with_voice_clone.hubert.hubert_manager import HuBERTManager
from bark_with_voice_clone.hubert.pre_kmeans_hubert import CustomHubert
from bark_with_voice_clone.hubert.customtokenizer import CustomTokenizer

global models
models = {}
model_devices = {}

#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')#'cpu'


def run_process(rank,voice_sublists,samples,override=None,skip_list=None):
    voices_list = voice_sublists[rank]
    voice_clone_samples(rank,samples,voices_list,override,skip_list)

def clone_voice_to_sample(sample,voice,model,hubert_model,tokenizer,device):
    global models
    global model_devices
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
    sample_audio, sample_sr = torchaudio.load(sample)
    sample_wav = convert_audio(sample_audio,sample_sr,model.sample_rate, model.channels).to(device)
    semantic_vectors_2 = hubert_model.forward(sample_wav,input_sample_hz=model.sample_rate)
    semantic_tokens_2 = tokenizer.get_token(semantic_vectors_2)
    semantic_tokens_2 = semantic_tokens_2.cpu().numpy()
    cloned_audio = semantic_to_waveform(semantic_tokens_2,history_prompt=voice_name)
    return cloned_audio, model.sample_rate


def voice_clone_samples_old(device_id,samples,voices_list, override=False, skip_list=None):
    device = torch.device(f"cuda:{device_id}")
    # we need model preloading here:
    global models
    global model_devices

    logging.info(f"Process on device {device} cloning voices from {voices_list[0]} to {voices_list[-1]}")
    model = load_codec_model(use_gpu=False).to(device)
    
    os.chdir('/project/')
    hubert_manager = HuBERTManager()
    hubert_manager.make_sure_hubert_installed()
    hubert_manager.make_sure_tokenizer_installed()
    
    hubert_model = CustomHubert(checkpoint_path='/project/data/models/hubert/hubert.pt',device=device).to(device)
    tokenizer = CustomTokenizer.load_from_checkpoint('/project/data/models/hubert/tokenizer.pth',map_location=device).to(device)
    for voice in voices_list:
        voice_name, _ = os.path.splitext(os.path.basename(voice))
        for sample in samples:
            filename = os.path.basename(sample)
            sample_name, ext = os.path.splitext(filename)
            cloned_path = os.path.join(ROOT_DIR,'audio','clone',f'{sample_name}_bark_{voice_name}{ext}')
            if f'{sample_name}_bark_{voice_name}{ext}' in skip_list:
                logging.info(f"File {cloned_path} is in skip list, skipping.")
                continue
            if os.path.isfile(cloned_path) and not override:
                logging.info(f"File {cloned_path} already exists, skipping. Use --override option to change the behavior.")
                continue
            logging.info(f"Processing voice cloning with Bark for sample {filename} with voice {voice_name} on device {device}")
            start = time.time()
            with torch.no_grad():
                cloned_audio, sr = clone_voice_to_sample(sample,voice,model,hubert_model,tokenizer,device)
            end = time.time()
            duration = end - start
            logging.info(f"Ended voice cloning with Bark for sample {filename} with voice {voice_name}, time: {duration}, device: {device}")
            sf.write(cloned_path,cloned_audio,sr)
    logging.info(f"Device {device} ended cloning.")

def voice_clone_samples(device_id,sample_voice_tuple_list, override=False, skip_list=None):
    device = torch.device(f"cuda:{device_id}")
    # we need model preloading here:
    global models
    global model_devices

    logging.info(f"Process on device {device} cloning voices from {voices_list[0]} to {voices_list[-1]}")
    model = load_codec_model(use_gpu=False).to(device)
    
    os.chdir('/project/')
    hubert_manager = HuBERTManager()
    hubert_manager.make_sure_hubert_installed()
    hubert_manager.make_sure_tokenizer_installed()
    
    hubert_model = CustomHubert(checkpoint_path='/project/data/models/hubert/hubert.pt',device=device).to(device)
    tokenizer = CustomTokenizer.load_from_checkpoint('/project/data/models/hubert/tokenizer.pth',map_location=device).to(device)
    for sample, voice in sample_voice_tuple_list:
        voice_name, _ = os.path.splitext(os.path.basename(voice))
        filename = os.path.basename(sample)
        sample_name, ext = os.path.splitext(filename)
        cloned_path = os.path.join(ROOT_DIR,'audio','clone',f'{sample_name}_bark_{voice_name}{ext}')
        if f'{sample_name}_bark_{voice_name}{ext}' in skip_list:
            logging.info(f"File {cloned_path} is in skip list, skipping.")
            continue
        if os.path.isfile(cloned_path) and not override:
            logging.info(f"File {cloned_path} already exists, skipping. Use --override option to change the behavior.")
            continue
        logging.info(f"Processing voice cloning with Bark for sample {filename} with voice {voice_name} on device {device}")
        start = time.time()
        with torch.no_grad():
            cloned_audio, sr = clone_voice_to_sample(sample,voice,model,hubert_model,tokenizer,device)
        end = time.time()
        duration = end - start
        logging.info(f"Ended voice cloning with Bark for sample {filename} with voice {voice_name}, time: {duration}, device: {device}")
        sf.write(cloned_path,cloned_audio,sr)
    logging.info(f"Device {device} ended cloning.")

def main(args):
    os.makedirs(os.path.join(ROOT_DIR,'audio','watermarked'),exist_ok=True)
    os.makedirs(os.path.join(ROOT_DIR, 'audio','clone'),exist_ok=True)
    if args.samples is not None:
        samples = parse_samples(args.samples)
    if args.voices is not None:
        voices = parse_samples(args.voices)
        #sublists = np.array_split(voices,num_devices)
    if args.detect is not None:
        samples_to_detect = parse_samples(args.detect)
    else:
        samples_to_detect = None
    if args.skip_list is not None:
        skip_list = parse_already_done(args.skip_list[0])
    else:
        skip_list = None
    all_combinations = [(sample,voice) for sample in samples for voice in voices]
    kept_combinations = filter_combinations(all_combinations,skip_list,'/project/audio/clone/')
    print(f"Skipped {len(all_combinations) - len(kept_combinations)} combinations")
    sublists = np.array_split(kept_combinations,num_devices)
    if args.device is not None:
        num_devices = torch.cuda.device_count()
        device_id = int(args.device)
        device = torch.device(f"cuda:{device_id}")
        assert 0 <= device_id <= num_devices, "Incorrect device id"
        global models
        global model_devices
        models["text"] = _load_model("/project/models/text_2.pt",device,model_type="text")
        models["text"]["model"].to(device)
        models["fine"] = _load_model("/project/models/fine_2.pt",device,model_type="fine")
        models["coarse"] = _load_model("/project/models/coarse_2.pt",device,model_type="coarse")
        model_devices["text"] = device
        model_devices["fine"] = device
        model_devices["coarse"] = device
        # very important!
        bark_with_voice_clone.bark.generation.models = models
        bark_with_voice_clone.bark.generation.model_devices = model_devices
    voice_clone_samples(device_id,sublists[device_id],args.override,skip_list)

def parse_already_done(filename):
    with open(filename,"r") as f:
        contents = f.read().splitlines()
    return contents
    
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

def filter_combinations(combinations,skip_list,clone_dir):
    created_file_list = os.listdir(clone_dir)
    created_file_basenames = [tuple(i.split('_bark_')) for i in created_file_list]
    created_file_basenames = [(i+'.flac',j) for i, j in created_file_basenames]
    #combinations = set(combinations)
    skip_basenames = [tuple(i.split('_bark_')) for i in skip_list]
    skip_basenames = [(i+'.flac',j) for i, j in skip_basenames]
    combinations_basenames = [(os.path.basename(i),os.path.basename(j)) for i,j in combinations]
    z1 = [(i,j) for i,j,k,l in zip(combinations,combination_basenames) if (k,l) not in created_file_basenames ]
    z2 = [(i,j) for i,j,k,l in zip(combinations,combination_basenames) if (k,l) not in skip_basenames ]
    z1 = set(z1)
    z2 = set(z2)
    z = z1.intersection(z2)
    return list(z)
    

def parse_samples(samples_list):
    file_list = []
    if samples_list is None:
        return []
    for sample in samples_list:
        if os.path.isdir(sample):
            for item in sorted(os.listdir(sample)):
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
    parser.add_argument("--skip_list",nargs='+',help="Input .txt file with list of permutations that should be skipped",required=False)
    parser.add_argument("--device",help="Number of CUDA device to run the models on.",required=False)
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
