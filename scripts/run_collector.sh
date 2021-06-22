#!/bin/bash

docker run -d --rm --name carla-client -v ~/cachefs:/home/carla-dataset-runner/data --network host -it carla-client /bin/bash
