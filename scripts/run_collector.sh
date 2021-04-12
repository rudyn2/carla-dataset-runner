#!/bin/bash

nvidia-docker run -v ~/cachefs/:/home/carla-dataset-runner/data --network host carla-dataset-collector -e DOCKER_HOST_IP=172.18.0.1 /bin/bash
