#FROM ubuntu:20.04
FROM debian:stable-slim
ENV PATH="/root/miniconda3/bin:${PATH}"
ARG PATH="/root/miniconda3/bin:${PATH}"
RUN apt-get update

RUN apt-get install -y wget && rm -rf /var/lib/apt/lists/*

RUN wget \
    https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
    && mkdir /root/.conda \
    && bash Miniconda3-latest-Linux-x86_64.sh -b \
    && rm -f Miniconda3-latest-Linux-x86_64.sh 
RUN apt-get update

# Create the environment:
COPY environment.yml .
RUN conda env create -f environment.yml
RUN apt-get install -y gnupg2
RUN echo "deb http://us.archive.ubuntu.com/ubuntu/ bionic main" >> /etc/apt/sources.list
RUN apt-key adv --keyserver keyserver.ubuntu.com --recv-keys 3B4FE6ACC0B21F32
RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6  -y
RUN apt-get install libjpeg-turbo8 -y

# Make RUN commands use the new environment:
SHELL ["conda", "run", "-n", "carla-data-collector", "/bin/bash", "-c"]
RUN pip install scikit-image
RUN pip install termcolor
RUN pip install wandb

# create important folders
RUN mkdir /home/carla-dataset-runner
RUN mkdir /home/carla-dataset-runner/PythonAPI
RUN mkdir /home/carla-dataset-runner/data
RUN mkdir /home/carla-dataset-runner/src
RUN mkdir /home/carla-dataset-runner/carla_egg

# copy the code
COPY carla_egg/carla-0.9.11-py3.7-linux-x86_64.egg /home/carla-dataset-runner/carla_egg/
COPY PythonAPI/ /home/carla-dataset-runner/PythonAPI/
COPY src/ /home/carla-dataset-runner/src/
COPY *.py /home/carla-dataset-runner/
WORKDIR /home/carla-dataset-runner

# Set some envirnonment variables
ENV PYTHONPATH "${PYTHONPATH}:/home/carla-dataset-runner/PythonAPI"
ENV PYTHONPATH "${PYTHONPATH}:/home/carla-dataset-runner/carla_egg/carla-0.9.11-py3.7-linux-x86_64.egg"

RUN ["conda", "init", "--all"]
# Run data collection, if logs aren't working, prefer to do it manually
# RUN python main.py test_docker -ve 100 -wa 50 -T 1000 -H 172.26.0.1 -t Town01 --width 288 --height 288 --routes /home/carla-dataset-runner/routes/routes_training.xml
