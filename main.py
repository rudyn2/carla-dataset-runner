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
import faulthandler
import os
import sys
import traceback

import settings
from CarlaWorld import CarlaWorld
from HDF5Saver import HDF5Saver
from JsonSaver import JsonSaver
from route_parser import RouteParser

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

faulthandler.enable()


def record_one_ego_run(world: CarlaWorld, vehicles: int, walkers: int, weather: list, frames: int, debug: bool,
                       route=None):
    world.set_ego_agent(route)
    world.spawn_npcs(number_of_vehicles=vehicles, number_of_walkers=walkers)
    world.set_weather(weather)

    media, info = world.begin_data_acquisition(sensor_width, sensor_height, fov,
                                               frames_to_record_one_ego=frames,
                                               debug=debug, route=route)

    world.remove_npcs()
    world.remove_sensors()
    return media, info


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Settings for the data capture",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('hdf5_file', default=None, type=str, help='name of hdf5 file to save the data')
    parser.add_argument('-H', '--host', default='localhost', type=str, help='CARLA server ip address')
    parser.add_argument('-p', '--port', default=2000, type=int, help='CARLA server port number')
    parser.add_argument('-n', default=1, type=int, help='number of ego executions')
    parser.add_argument('-T', default=100, type=int,
                        help='number of frames to record per ego execution')
    parser.add_argument('-ea', '--early-break', type=int, help='Early break at some number of routes recorded')
    parser.add_argument('-t', '--town', default='Town01', type=str, help="town to use")
    parser.add_argument('-wi', '--width', default=1024, type=int, help="camera rgb and depth sensor width in pixels")
    parser.add_argument('-he', '--height', default=768, type=int, help="camera rgb and depth sensor width in pixels")
    parser.add_argument('-ve', '--vehicles', default=0, type=int, help="number of vehicles to spawn in the simulation")
    parser.add_argument('-wa', '--walkers', default=0, type=int, help="number of walkers to spawn in the simulation")
    parser.add_argument('-r', '--routes', default=None, type=str, help="path to xml file with predefined routes")
    parser.add_argument('--debug', action="store_true")

    args = parser.parse_args()
    assert (args.hdf5_file is not None)
    assert (args.width > 0 and args.height > 0)
    if args.vehicles == 0 and args.walkers == 0:
        print('Are you sure you don\'t want to spawn vehicles and pedestrians in the map?')

    try:
        host_ip = os.environ['DOCKER_HOST_IP']
    except KeyError:
        host_ip = args.host

    # Sensor setup (rgb and depth share these values)
    # 1024 x 768 or 1920 x 1080 are recommended values. Higher values lead to better graphics but larger filesize
    sensor_width = args.width
    sensor_height = args.height
    fov = 100

    # Beginning data capture procedure
    json_file = JsonSaver(os.path.join("data", args.hdf5_file + ".json"))
    print("HDF5 File opened")
    CarlaWorld = CarlaWorld(town=args.town, host=host_ip, port=args.port)
    weather_lookup = CarlaWorld.weather_lookup

    timestamps = []
    counter = 0
    print('Starting to record data...\n')

    if args.routes is None:
        for run in range(args.n):

            for weather_id, weather_option in enumerate(CarlaWorld.weather_options):

                _id = f"run_{str(counter).zfill(3)}_{weather_lookup[weather_id]}"
                print(f"RUN ID: {_id}")
                print(f"\nWeather: {weather_lookup[weather_id]}")
                not_saved = True
                while not_saved:
                    try:
                        media_data, info_data = record_one_ego_run(CarlaWorld, args.vehicles, args.walkers,
                                                                   weather_option, args.T, args.debug)
                        hdf5_file = HDF5Saver(sensor_width, sensor_height,
                                              os.path.join("data", args.hdf5_file + ".hdf5"))
                        hdf5_file.save_one_ego_run(media_data=media_data, run_id=_id)
                        hdf5_file.close_hdf5()
                        json_file.save_one_ego_run(info_data=info_data, run_id=_id)
                        not_saved = False

                    except AttributeError as e:
                        print("CARLA Server side error. Restarting...\n")
                    except Exception as e:
                        traceback.print_exc(e)

                    CarlaWorld.reset()
                print("--------------------------------------")

            counter += 1
    else:
        print("Reading XML Routes")
        parser = RouteParser(CarlaWorld.map, str(args.routes))
        routes = parser.parse_file()
        hop_resolution = 1.0

        for route_id in list(routes.keys()):
            if routes[route_id]['town'] == args.town:
                CarlaWorld.world_tag = routes[route_id]['town']
                CarlaWorld.reset()

                for weather_id, weather_option in enumerate(CarlaWorld.weather_options[:1]):

                    _id = f"run_{str(counter).zfill(3)}_{weather_lookup[weather_id]}"
                    print(f"RUN ID: {_id}")
                    print(f"\nWeather: {weather_lookup[weather_id]}")
                    not_saved = True
                    while not_saved:
                        try:
                            media_data, info_data = record_one_ego_run(CarlaWorld, args.vehicles, args.walkers,
                                                                       weather_option, args.T, args.debug,
                                                                       routes[route_id]['waypoints'])
                            print("Saving")
                            hdf5_file = HDF5Saver(sensor_width, sensor_height,
                                                  os.path.join("data", args.hdf5_file + ".hdf5"))
                            hdf5_file.save_one_ego_run(media_data=media_data, run_id=_id)
                            hdf5_file.close_hdf5()
                            print("Images saved!")
                            json_file.save_one_ego_run(info_data=info_data, run_id=_id)
                            print("Metadata saved!")
                            not_saved = False

                        except AttributeError as e:
                            print(e)
                            print("CARLA Server side error. Restarting...\n")
                        except Exception as e:
                            traceback.print_exc(e)

                        CarlaWorld.reset()
                    print("--------------------------------------")
                counter += 1

                if counter == args.early_break:
                    print("Early stopped!")
                    break

    print("\n\nData recording has finished successfully.")
