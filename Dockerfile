FROM conda/miniconda3
RUN conda init bash
COPY env.yml ./
RUN conda env create -f env.yml
SHELL ["bash", "-c"]
# RUN conda activate audio-watermark

