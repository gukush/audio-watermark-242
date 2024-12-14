# audio-watermark-242
Repository for research project about watermarkng audio
## Setup
This repo requires some submodules, clone it recursively:
```
git clone --recursive https://github.com/gukush/audio-watermark-242.git
```
To run the sample experiment first create image using docker.
From repository's root directory run (for now Dockerfile-test is the better dockerfile):
```
docker build -t audio-watermark -f Dockerfile-test .
```
Then create the container (also from repository's root directory), on Windows please replace the source=./ part with source={C:\\absolute\\path\\to\\repo\\audio-watermark-242\\}.
```
docker run -it --mount type=bind,source=./,target=/project/ --name audio-watermark audio-watermark
```
When you have CUDA container toolkit installed then you can add GPUs in by adding *--runtime nvidia --gpus all* options.
When you want to access the container from other terminal window, run (on Linux):
```
docker exec -it audio-watermark /bin/bash
```
Finally, to see sample workflow run following command
```
cd /project && python simple_example.py
```
This will produce a result in form of sample audio of watermarked voice that was cloned (accessible in audio folder).

## Other Info
Results of NISQA predictions are in results/NISQA_results.csv

## Known issues
Long setup times - the image after building is around 10GB. Lots of hard-coded paths in files.
