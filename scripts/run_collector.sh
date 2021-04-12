#!/bin/bash

nvidia-docker run -v ~/cachefs/:/home/carla-dataset-runner/data -e DOCKER_HOST_IP="172.18.0.1" --network host carla-dataset-collector /bin/bash
