import os
import subprocess
from multiprocessing import Pool
import torch
import argparse

def voice_clone_task(task):
    """
    Task for voice cloning.
    Each task is a tuple: (speaker_file, reference_file, output_file, device)
    """
    speaker_file, reference_file, output_file, device = task
    print(f"Processing: {output_file}")
    # Set device for this subprocess
    if device.startswith("cuda"):
        os.environ["CUDA_VISIBLE_DEVICES"] = device.split(":")[1]
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""

    try:
        result = subprocess.run(
            [
                'tts', 
                '--model_name', 'tts_models/multilingual/multi-dataset/your_tts',
                '--speaker_wav', speaker_file,
                '--reference_wav', reference_file,
                '--language_idx', 'en',
                '--out_path', output_file
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        print(f"Completed: {output_file}")
        print(result.stdout.decode('utf-8'))
    except Exception as e:
        print(f"Error processing {output_file}: {e}")

def main():
    parser = argparse.ArgumentParser(description="Parallel Voice Cloning with YourTTS")
    parser.add_argument('--speaker_dir', type=str, required=True, help="Path to the directory containing speaker WAV files")
    parser.add_argument('--reference_dir', type=str, required=True, help="Path to the directory containing reference WAV files")
    parser.add_argument('--output_dir', type=str, required=True, help="Path to the directory to save output WAV files")
    args = parser.parse_args()
    # Make sure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)

    # Get sorted lists of files
    speaker_files = sorted([os.path.join(args.speaker_dir, f) for f in os.listdir(args.speaker_dir) if f.endswith('.flac')])
    reference_files = sorted([os.path.join(args.reference_dir, f) for f in os.listdir(args.reference_dir) if f.endswith('.flac')])
    # Determine available devices
    if torch.cuda.is_available():
        devices = [f"cuda:{i}" for i in range(torch.cuda.device_count())]
    else:
        devices = ["cpu"]

    # Prepare tasks (many-to-many mapping)
    tasks = [
        (
            speaker_file,
            reference_file,
            os.path.join(args.output_dir, f"{os.path.splitext(os.path.basename(speaker_file))[0]}_{os.path.splitext(os.path.basename(reference_file))[0]}.flac"),
            devices[idx % len(devices)]  # Assign device in a round-robin manner
        )
        for idx, (speaker_file, reference_file) in enumerate(
            [(s, r) for s in speaker_files for r in reference_files]
        )
    ]

    with Pool(processes=len(devices)) as pool:
        pool.map(voice_clone_task, tasks)

if __name__ == "__main__":
    main()