FROM continuumio/miniconda3
#RUN conda install -n base conda-libmamba-solver
#RUN conda config --set solver libmamba
RUN conda init bash
#COPY env-nisqa.yml ./
COPY env-main.yml ./
#RUN conda env create -f env-nisqa.yml
RUN conda env create -f env-main.yml
SHELL ["bash", "-c"]
# RUN conda activate audio-watermark
ADD /audioseal /audioseal
WORKDIR /project
RUN conda activate audio-watermark-main && pip install -e ./audioseal
 #- might be used as workaround

