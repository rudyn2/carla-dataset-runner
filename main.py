"""
Dataset creation plan
200 frames = 1,7 GB
Total planned: 20.000 frames = 170 GB
5 Towns
5 Weathers
x frames per ego vehicle
y amount of ego vehicles

Total frames planned = x * y * 5 weathers * 5 towns
20.000 / 5 towns = x * y * 5 weathers
x * y = 800
if x = 60 frames per ego, then
60 * y = 800
y ~= 13 egos

i.e., 13 egos * 60 frames * 5 weathers = 3900 frames per town
13900 * 5 towns = 19500 frames total
19500 frames ~= 165.75 GB

Suggested amount of vehicles and walkers so that traffic jam occurence is minimized
Town01 - 100 vehic 200 walk
Town02 - 50 vehic 100 walk
Town03 - 200 vehic 150 walk
Town04 - 250 vehic 100 walk
Town05 - 150 vehic 150 walk
"""

import argparse
import os
import sys
import pathlib
import time

from src.engine import CarlaExtractor
from src.utils.HDF5Saver import HDF5Saver
from src.utils.JsonSaver import JsonSaver
from src.utils.WeatherSelector import WeatherSelector
from termcolor import colored
import uuid


# TODO: Add main routine for data extraction.
# TODO: Implement cloud  logging (wandb could be a good option)
def run(args_):

    hdf5_file_path = str(pathlib.Path(args_.hdf5_file).absolute()) + ".hdf5"
    json_file_path = str(pathlib.Path(args_.hdf5_file).absolute()) + ".json"
    print(colored(f"Host: {args_.host}\nPort: {args_.port}\nSensor width: {args_.width}"
                  f"\nSensor height: {args_.height}\nTown: {args_.town}\nVehicles: {args_.vehicles}"
                  f"\nWalkers: {args_.walkers}\nHDF5 output path: {hdf5_file_path}\nJSON output path: {json_file_path}", "magenta"))

    engine = CarlaExtractor(host=args_.host,
                            port=args_.port,
                            sensor_width=args_.width,
                            sensor_height=args_.height,
                            town=args_.town)
    hdf5_saver = HDF5Saver(sensor_width=args_.width,
                           sensor_height=args_.height,
                           file_path_to_save=hdf5_file_path)
    json_saver = JsonSaver(json_file_path)

    weather_selector = WeatherSelector()
    weather_hooks = [weather_selector.morning, weather_selector.midday, weather_selector.default,
                     weather_selector.almost_night, weather_selector.afternoon]

    try:
        print(colored("Starting the extraction...", "cyan"))
        for iteration in range(args_.n):
            for weather in weather_hooks:
                print(colored(f"weather: {weather.__name__}", "white"))
                engine.set_weather(weather())
                run_id = str(uuid.uuid4())
                media, meta = engine.record(vehicles=args_.vehicles,
                                            walkers=args_.walkers,
                                            max_frames=args_.T,
                                            debug=args_.debug)
                hdf5_saver.save_one_ego_run(run_id=run_id, media_data=media)
                json_saver.save_one_ego_run(run_id=run_id, info_data=meta)

        print(colored("Extraction finished!", "cyan"))

    finally:
        hdf5_saver.close_hdf5()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Settings for the data capture",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('hdf5_file', default=None, type=str, help='name of hdf5 file where data will be stored')
    parser.add_argument('-H', '--host', default='localhost', type=str, help='CARLA server ip address')
    parser.add_argument('-p', '--port', default=2000, type=int, help='CARLA server port number')
    parser.add_argument('-n', default=1, type=int, help='number of ego executions')
    parser.add_argument('-T', default=100, type=int,
                        help='number of frames to record per ego execution')
    parser.add_argument('-t', '--town', default='Town01', type=str, help="town to use")
    parser.add_argument('-wi', '--width', default=1024, type=int, help="camera rgb and depth sensor width in pixels")
    parser.add_argument('-he', '--height', default=768, type=int, help="camera rgb and depth sensor width in pixels")
    parser.add_argument('-ve', '--vehicles', default=0, type=int, help="number of vehicles to spawn in the simulation")
    parser.add_argument('-wa', '--walkers', default=0, type=int, help="number of walkers to spawn in the simulation")
    parser.add_argument('--debug', action="store_true")

    args = parser.parse_args()
    run(args)
