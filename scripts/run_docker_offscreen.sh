#!/bin/bash

docker run --name carla-data-collector -d --rm -p 2000-2002:2000-2002 --runtime=nvidia -it carlasim/carla:0.9.11 /bin/bash -c 'SDL_VIDEODRIVER=offscreen ./CarlaUE4.sh -opengl -quality-level=low -world-port=2000'