# audio-watermark-242
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