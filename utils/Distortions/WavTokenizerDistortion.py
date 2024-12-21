import os
import sys
sys.path.append('/project/WavTokenizer')
from encoder.utils import convert_audio
import torchaudio
import torch
from decoder.pretrained import WavTokenizer

config_path = "/project/models/wavtokenizer/wavtokenizer_mediumdata_frame75_3s_nq1_code4096_dim512_kmeans200_attn.yaml"
model_path = "/project/models/wavtokenizer/wavtokenizer_medium_speech_320_24k_v2.ckpt"
global wavtokenizer

def distort_with_wavtokenizer(sample,device,override):
    global wavtokenizer
    filename = os.path.basename(sample)
    sample_name, ext = os.path.splitext(filename)
    distorted_path = os.path.join(ROOT_DIR,'audio','distorted',f'{sample_name}_wavtokenizer{ext}')
    if os.path.isfile(distorted_path) and not override:
        logging.info(f"File {distorted_path} already exists, skipping. Use --override option to change the behavior.")
        continue
    audio, sr = torchaudio.load(sample)
    audio = convert_audio(audio, sr, 24000, 1)
    bandwidth_id = torch.tensor([0])
    audio = audio.to(device)
    with torch.no_grad():
        sth_tmp, discrete_code = wavtokenizer.encode_infer(audio,bandwidth_id=bandwidth_id)
    features = wavtokenizer.codes_to_features(discrete_code)
    audio_out = wavtokenizer.decode(features, bandwidth_id=bandwidth_id)
    torchaudio.save(audio_out_path,audio_out,24000)
#torch.save(discrete_code,"/project/tmp/Elevenlabs_latent_wavtokenizer.pth")

bandwidth_id = torch.tensor([0])

audio_out_path = "/project/tmp/test_clone_wavtokenizer.mp3" #Elevenlabs_184339_snippet2


def main(args):
    global wavtokenizer
    num_devices = torch.cuda.device_count()
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
    if args.gpus is not None:
        num_devices = int(args.gpus)
    if len(kept_combinations) < 12:
        print("Less than 12 elements left, doing all on same device without splitting")
        sublists = [kept_combinations] * num_devices
    else:
        sublists = np.array_split(kept_combinations,num_devices)
    if args.device is not None:
        if args.device == 'cpu':
            device = 'cpu'
            device_id = 'cpu'
        else:
            device_id = int(args.device)
            device = f"cuda:{device_id}"#torch.device(f"cuda:{device_id}")
            assert 0 <= device_id <= num_devices, "Incorrect device id"
        wavtokenizer = WavTokenizer.from_pretrained0802(config_path,model_path)
        wavtokenizer = wavtokenizer.to(device)
    if device == 'cpu':
        voice_clone_samples(device_id,sublists[0],args.override,skip_list)
    else:
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
    created_file_basenames = [tuple(i.split('_openvoice_')) for i in created_file_list]
    created_file_basenames = [i for i in created_file_basenames if len(i) > 1]
    created_file_basenames = [(i+'.flac',j) for i, j in created_file_basenames]
    #combinations = set(combinations)
    skip_basenames = [tuple(i.split('_openvoice_')) for i in skip_list]
    skip_basenames = [i for i in skip_basenames if len(i) > 1]
    skip_basenames = [(i+'.flac',j) for i, j in skip_basenames]
    combinations_basenames = [(os.path.basename(i),os.path.basename(j)) for i,j in combinations]
    z = []
    for ((i,j),(k,l)) in zip(combinations,combinations_basenames):
        if (k,l) not in created_file_basenames:
            if (k,l) not in skip_basenames:
                z.append((i,j))
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
    parser.add_argument("--skip_list",nargs='+',help="Input .txt file with list of permutations that should be skipped",required=False)
    parser.add_argument("--device",help="Number of CUDA device to run the models on.",required=False)
    parser.add_argument("--gpus",help="Total number of gpus to run on (default is all available)",required=False)
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
