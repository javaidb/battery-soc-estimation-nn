FROM pytorch/pytorch:2.4.0-cuda12.4-cudnn9-runtime

RUN apt-get update

WORKDIR /app

# Copy the requirements.txt into the container
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

ENV NVIDIA_VISIBLE_DEVICES=all