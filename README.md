# audio-watermark-242
## Setup
Repository for research project about watermarkng audio

To run the sample experiment first create image using docker.
From repository's root directory run:
```
docker build -t audio-watermark .
```
Then create the container:
```
docker run -it --mount type=bind,source=./,target=/project/ --name audio-watermark audio-watermark
```
When you want to access the container from other terminal window, run (on Linux):
```
docker exec -it audio-watermark /bin/bash
```
Finally, in the docker container run following to activate conda environment:
```
cd /project && conda activate audio-watermark
```

## Known issues
Torchaudio requires proper version of ffmpeg!!!
Long setup times - the image after building is around 7GB. Solving conda environment takes some time. So in case of changes of env.yml it should be advised to install packages without reseting the image (either through conda or pip).
To check current environment used in conda, run following command:
```
conda env export --name audio-watermark > env-compare.yml.tmp
```
For some reason gdown from env.yml was not installed during docker build.
```
pip install gdown
```