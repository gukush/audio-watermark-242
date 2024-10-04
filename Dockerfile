FROM python:3.12.6-bookworm
RUN apt update
RUN apt install -y ffmpeg
COPY requirements.txt /project/
WORKDIR /project
RUN pip install -r requirements.txt
