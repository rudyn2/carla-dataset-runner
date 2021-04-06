"""
Alan Naoto
Created: 14/10/2019

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
import traceback

import settings
from CarlaWorld import CarlaWorld
from HDF5Saver import HDF5Saver
from JsonSaver import JsonSaver

initial_path = set(sys.path)
sys.path.append(settings.CARLA_EGG_PATH)

# ADD
try:
    sys.path.append(os.path.abspath('.') + '/PythonAPI/carla')
except IndexError:
    pass

new_paths = set(sys.path) - initial_path
for path in new_paths:
    print(f"Added: {path} to the Path")


def record_one_ego_run(world: CarlaWorld, vehicles: int, walkers: int, weather: list, frames: int, debug: bool):
    world.spawn_npcs(number_of_vehicles=vehicles, number_of_walkers=walkers)
    world.set_weather(weather)

    media, info = world.begin_data_acquisition(sensor_width, sensor_height, fov,
                                               frames_to_record_one_ego=frames,
                                               debug=debug)

    world.remove_npcs()
    world.remove_sensors()
    return media, info


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Settings for the data capture",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('hdf5_file', default=None, type=str, help='name of hdf5 file to save the data')
    parser.add_argument('-wi', '--width', default=1024, type=int, help="camera rgb and depth sensor width in pixels")
    parser.add_argument('-he', '--height', default=768, type=int, help="camera rgb and depth sensor width in pixels")
    parser.add_argument('-ve', '--vehicles', default=0, type=int, help="number of vehicles to spawn in the simulation")
    parser.add_argument('-wa', '--walkers', default=0, type=int, help="number of walkers to spawn in the simulation")
    parser.add_argument('--debug', action="store_true")
    parser.add_argument('-v', '--video', action="store_true",
                        help="record a mp4 video on top of the recorded hdf5 file")
    parser.add_argument('-d', '--depth', action='store_true', help="show the depth video side by side with the rgb")
    args = parser.parse_args()
    assert (args.hdf5_file is not None)
    assert (args.width > 0 and args.height > 0)
    if args.vehicles == 0 and args.walkers == 0:
        print('Are you sure you don\'t want to spawn vehicles and pedestrians in the map?')

    # Sensor setup (rgb and depth share these values)
    # 1024 x 768 or 1920 x 1080 are recommended values. Higher values lead to better graphics but larger filesize
    sensor_width = args.width
    sensor_height = args.height
    fov = 90

    # Beginning data capture procedure
    HDF5_file = HDF5Saver(sensor_width, sensor_height, os.path.join("data", args.hdf5_file + ".hdf5"))
    json_file = JsonSaver(os.path.join("data", args.hdf5_file + ".json"))
    print("HDF5 File opened")
    CarlaWorld = CarlaWorld(hdf5_file=HDF5_file)
    weather_lookup = CarlaWorld.weather_lookup

    timestamps = []
    runs = 2
    frames_per_ego_run = 50
    print('Starting to record data...\n')
    for run in range(runs):
        counter = 0
        for weather_id, weather_option in enumerate(CarlaWorld.weather_options):

            _id = f"run_{str(counter).zfill(3)}_{weather_lookup[weather_id]}"
            print(f"RUN ID: {_id}")
            print(f"\nWeather: {weather_lookup[weather_id]}")
            not_saved = True
            while not_saved:
                try:
                    media_data, info_data = record_one_ego_run(CarlaWorld, args.vehicles, args.walkers,
                                                               weather_option, frames_per_ego_run, args.debug)
                    HDF5_file.save_one_ego_run(media_data=media_data, run_id=_id)
                    json_file.save_one_ego_run(info_data=info_data, run_id=_id)
                    not_saved = False

                except AttributeError as e:
                    print("CARLA Server side error. Restarting...\n")
                except Exception as e:
                    traceback.print_exc(e)

                CarlaWorld.reset()
            print("--------------------------------------")

    print("\n\nData recording has finished successfully.")
