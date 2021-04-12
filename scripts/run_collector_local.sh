#!/bin/bash

docker run -v ~/Documents/carla-dataset-runner/data/:/home/carla-dataset-runner/data -e DOCKER_HOST_IP="172.26.0.1" --network host carla-dataset-collector /bin/bash
