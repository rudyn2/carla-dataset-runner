#!/bin/bash

docker build --network host --rm -t test_collector . && docker run -v ~/cachefs:/home/carla-dataset-runner/data test_collector /bin/bash
#
