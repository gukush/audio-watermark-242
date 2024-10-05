FROM python:3.12.6-bookworm
RUN apt update
RUN apt install -y ffmpeg curl
COPY requirements.txt /project/
COPY audioseal /project/audioseal
COPY OpenVoice /project/OpenVoice
COPY SilentCipher /project/SilentCipher
WORKDIR /project
RUN pip install -r requirements.txt
RUN curl -o checkpoints.zip https://myshell-public-repo-host.s3.amazonaws.com/openvoice/checkpoints_v2_0417.zip \
    && unzip checkpoints.zip -d /models \
    && rm checkpoints.zip
WORKDIR /project/SilentCipher
RUN python -m build
RUN pip install dist/*.whl
