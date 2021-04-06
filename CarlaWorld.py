import os
import sys

import settings

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

import carla
import random
import time
import numpy as np
import cv2
from tqdm import tqdm

from spawn_npc import NPCClass
from set_synchronous_mode import CarlaSyncMode
from WeatherSelector import WeatherSelector

from agents.navigation.behavior_agent import BehaviorAgent


def parse_control(c):
    """
    Parse a carla.VehicleControl to a json object.
    """
    return {
        "brake": c.brake,
        "gear": c.gear,
        "hand_brake": c.hand_brake,
        "manual_gear_shift": c.manual_gear_shift,
        "reverse": c.reverse,
        "steer": c.steer,
        "throttle": c.throttle
    }


class CarlaWorld:
    def __init__(self, hdf5_file, world='Town02'):
        self.HDF5_file = hdf5_file
        self.world_tag = world

        # Carla initialization
        self.client = carla.Client('localhost', 2000)
        self.client.set_timeout(20.0)
        self.client.load_world(world)
        self.world = self.client.get_world()

        print('Successfully connected to CARLA')
        self.blueprint_library = self.world.get_blueprint_library()
        self.map = self.world.get_map()

        # Sensors stuff
        self.camera_x_location = 1.0
        self.camera_y_location = 0.0
        self.camera_z_location = 2.0
        self.sensors_list = []
        # Weather stuff
        self.weather_options = WeatherSelector().get_weather_options()  # List with weather options
        self.weather_lookup = WeatherSelector().get_weather_lookup()

        # Recording stuff
        self.total_recorded_frames = 0
        self.first_time_simulating = True

    def set_weather(self, weather_option):
        # Changing weather https://carla.readthedocs.io/en/stable/carla_settings/
        # Weather_option is one item from the list self.weather_options, which contains a list with the parameters
        weather = carla.WeatherParameters(*weather_option)
        self.world.set_weather(weather)

    def reset(self):
        self.client.load_world(self.world_tag)

    def remove_npcs(self):
        print('Destroying actors...')
        self.NPC.remove_npcs()
        print('Done destroying actors.')

    def spawn_npcs(self, number_of_vehicles, number_of_walkers):
        self.NPC = NPCClass()
        self.vehicles_list, _ = self.NPC.create_npcs(number_of_vehicles, number_of_walkers)

    def put_rgb_sensor(self, vehicle, sensor_width=640, sensor_height=480, fov=110):
        # https://carla.readthedocs.io/en/latest/cameras_and_sensors/
        bp = self.blueprint_library.find('sensor.camera.rgb')
        # bp.set_attribute('enable_postprocess_effects', 'True')  # https://carla.readthedocs.io/en/latest/bp_library/
        bp.set_attribute('image_size_x', f'{sensor_width}')
        bp.set_attribute('image_size_y', f'{sensor_height}')
        bp.set_attribute('fov', f'{fov}')

        # Adjust sensor relative position to the vehicle
        spawn_point = carla.Transform(carla.Location(x=self.camera_x_location, z=self.camera_z_location))
        self.rgb_camera = self.world.spawn_actor(bp, spawn_point, attach_to=vehicle)
        self.rgb_camera.blur_amount = 0.0
        self.rgb_camera.motion_blur_intensity = 0
        self.rgb_camera.motion_max_distortion = 0

        # Camera calibration
        calibration = np.identity(3)
        calibration[0, 2] = sensor_width / 2.0
        calibration[1, 2] = sensor_height / 2.0
        calibration[0, 0] = calibration[1, 1] = sensor_width / (2.0 * np.tan(fov * np.pi / 360.0))
        self.rgb_camera.calibration = calibration  # Parameter K of the camera
        self.sensors_list.append(self.rgb_camera)
        return self.rgb_camera

    def put_depth_sensor(self, vehicle, sensor_width=640, sensor_height=480, fov=110):
        # https://carla.readthedocs.io/en/latest/cameras_and_sensors/
        bp = self.blueprint_library.find('sensor.camera.depth')
        bp.set_attribute('image_size_x', f'{sensor_width}')
        bp.set_attribute('image_size_y', f'{sensor_height}')
        bp.set_attribute('fov', f'{fov}')

        # Adjust sensor relative position to the vehicle
        spawn_point = carla.Transform(carla.Location(x=self.camera_x_location, z=self.camera_z_location))
        self.depth_camera = self.world.spawn_actor(bp, spawn_point, attach_to=vehicle)
        self.sensors_list.append(self.depth_camera)
        return self.depth_camera

    def put_semantic_sensor(self, vehicle, semantic_width=640, semantic_height=480, fov=110):
        bp = self.blueprint_library.find('sensor.camera.semantic_segmentation')
        bp.set_attribute('image_size_x', f'{semantic_width}')
        bp.set_attribute('image_size_y', f'{semantic_height}')
        bp.set_attribute('fov', f'{fov}')

        # Adjust sensor relative position to the vehicle
        spawn_point = carla.Transform(carla.Location(x=self.camera_x_location, z=self.camera_z_location))
        self.semantic_camera = self.world.spawn_actor(bp, spawn_point, attach_to=vehicle)
        self.sensors_list.append(self.semantic_camera)
        return self.semantic_camera

    @staticmethod
    def process_depth_data(data, sensor_width, sensor_height):
        """
        normalized = (R + G * 256 + B * 256 * 256) / (256 * 256 * 256 - 1)
        in_meters = 1000 * normalized
        """
        data = np.array(data.raw_data)
        data = data.reshape((sensor_height, sensor_width, 4))
        data = data.astype(np.float32)
        # Apply (R + G * 256 + B * 256 * 256) / (256 * 256 * 256 - 1).
        normalized_depth = np.dot(data[:, :, :3], [65536.0, 256.0, 1.0])
        normalized_depth /= 16777215.0  # (256.0 * 256.0 * 256.0 - 1.0)
        depth_meters = normalized_depth * 1000
        return depth_meters

    @staticmethod
    def process_rgb_img(img, sensor_width, sensor_height):
        img = np.array(img.raw_data)
        img = img.reshape((sensor_height, sensor_width, 4))
        img = img[:, :, :3]  # taking out opacity channel
        return img

    @staticmethod
    def process_semantic_img(img, sensor_width, sensor_height):
        img = np.array(img.raw_data)
        img = img.reshape((sensor_height, sensor_width, 4))
        img = img[:, :, 2]  # taking just the RED channel
        return img

    def remove_sensors(self):
        for sensor in self.sensors_list:
            sensor.destroy()
        self.sensors_list = []

    def set_ego_agent(self, vehicle):

        ego_agent = BehaviorAgent(vehicle)
        spawn_points = self.map.get_spawn_points()
        random.shuffle(spawn_points)
        if spawn_points[0].location != ego_agent.vehicle.get_location():
            destination = spawn_points[0].location
        else:
            destination = spawn_points[1].location
        ego_agent.set_destination(ego_agent.vehicle.get_location(), destination, clean=True)
        ego_agent.update_information()
        print("Behavior Agent has been set up successfully")
        return ego_agent

    def begin_data_acquisition(self, sensor_width, sensor_height, fov, frames_to_record_one_ego=300,
                               debug: bool = False):
        """
        Records data using one ego for 'frames_to_record_one_ego' frames.
        """

        print("Beginning new data acquisition")
        # Changes the ego vehicle to be put the sensor
        current_ego_recorded_frames = 0
        media_data = []
        info_data = []
        # These vehicles are not considered because the cameras get occluded without changing their absolute position
        ego_vehicle = random.choice([x for x in self.world.get_actors().filter("vehicle.*") if x.type_id not in
                                     ['vehicle.audi.tt', 'vehicle.carlamotors.carlacola', 'vehicle.volkswagen.t2']])

        ego_agent = self.set_ego_agent(ego_vehicle)

        print(f"Using ego: {ego_vehicle.type_id}")
        self.put_rgb_sensor(ego_vehicle, sensor_width, sensor_height, fov)
        self.put_depth_sensor(ego_vehicle, sensor_width, sensor_height, fov)
        self.put_semantic_sensor(ego_vehicle, sensor_width, sensor_height, fov)

        # Begin applying the sync mode
        with CarlaSyncMode(self.world, self.rgb_camera, self.depth_camera, self.semantic_camera, fps=20) as sync_mode:
            # Skip initial frames where the car is being put on the ambient
            if self.first_time_simulating:
                for _ in range(30):
                    sync_mode.tick_no_data()

            progress_bar = tqdm(total=frames_to_record_one_ego)
            while True:
                if current_ego_recorded_frames == frames_to_record_one_ego:
                    progress_bar.close()
                    print(f"Recorded {current_ego_recorded_frames} frames for actual ego")
                    self.remove_sensors()
                    return media_data, info_data

                # Advance the simulation and wait for the data
                # Skip every nth frame for data recording, so that one frame is not that similar to another
                wait_frame_ticks = 0
                while wait_frame_ticks < 5:
                    sync_mode.tick_no_data()

                    wait_frame_ticks += 1
                    ego_agent.run_step()
                    ego_agent.update_information()

                # get data from sensors
                _, rgb_data, depth_data, semantic_data = sync_mode.tick(timeout=2.0)

                # Processing raw data
                rgb_array = self.process_rgb_img(rgb_data, sensor_width, sensor_height)
                depth_array = self.process_depth_data(depth_data, sensor_width, sensor_height)
                semantic_array = self.process_semantic_img(semantic_data, sensor_width, sensor_height)

                if debug:
                    cv2.imshow('rgb', rgb_array)
                    cv2.waitKey(10)

                timestamp = round(time.time() * 1000.0)
                ego_info = ego_agent.run_step_with_info()
                ego_vehicle.apply_control(ego_info["control"])
                ego_agent.update_information()

                # save data
                ego_info["control"] = parse_control(ego_info["control"])
                media_data.append({
                    "timestamp": timestamp,
                    "rgb": rgb_array,
                    "depth": depth_array,
                    "semantic": semantic_array
                })
                info_data.append({
                    "timestamp": timestamp,
                    "metadata": ego_info
                })
                current_ego_recorded_frames += 1

                if ego_info["at_tl"]:
                    s = f'Frame {current_ego_recorded_frames} | {ego_info["command"]} | ' \
                        f'Light state: {ego_info["tl_state"]} | Distance to TL: {ego_info["tl_distance"]:.3f} ' \
                        f'| Speed: {ego_info["speed"]:.2f} | Distance to center: {ego_info["lane_distance"]:.3f} ' \
                        f'| Orientation: {ego_info["lane_orientation"]:.3f}'
                else:
                    s = f'Frame {current_ego_recorded_frames} | {ego_info["command"]} | ' \
                        f'Speed: {ego_info["speed"]:.2f} | Distance to center: {ego_info["lane_distance"]:.3f} ' \
                        f'| Orientation: {ego_info["lane_orientation"]:.3f}'
                progress_bar.update()
                progress_bar.set_description(s)

    # def begin_data_acquisition2(self, sensor_width, sensor_height, fov, frames_to_record_one_ego=10, timestamps=[],
    #                            egos_to_run=1):
    #     print("Beginning new data acquisition")
    #     # Changes the ego vehicle to be put the sensor
    #     current_ego_recorded_frames = 0
    #     # These vehicles are not considered because the cameras get occluded without changing their absolute position
    #     ego_vehicle = random.choice([x for x in self.world.get_actors().filter("vehicle.*") if x.type_id not in
    #                                  ['vehicle.audi.tt', 'vehicle.carlamotors.carlacola', 'vehicle.volkswagen.t2']])
    #     print(f"Using ego: {ego_vehicle.type_id}")
    #     self.put_rgb_sensor(ego_vehicle, sensor_width, sensor_height, fov)
    #     self.put_depth_sensor(ego_vehicle, sensor_width, sensor_height, fov)
    #
    #     self.put_semantic_sensor(ego_vehicle, sensor_width, sensor_height, fov)
    #
    #     # Begin applying the sync mode
    #     with CarlaSyncMode(self.world, self.rgb_camera, self.depth_camera, self.semantic_camera, fps=30) as sync_mode:
    #         # Skip initial frames where the car is being put on the ambient
    #         if self.first_time_simulating:
    #             for _ in range(30):
    #                 sync_mode.tick_no_data()
    #
    #         while True:
    #             if current_ego_recorded_frames == frames_to_record_one_ego:
    #                 print(f"\nRecorded {current_ego_recorded_frames} frames for actual ego")
    #                 print('------------\n')
    #                 self.remove_sensors()
    #                 return timestamps
    #             # Advance the simulation and wait for the data
    #             # Skip every nth frame for data recording, so that one frame is not that similar to another
    #             wait_frame_ticks = 0
    #             while wait_frame_ticks < 5:
    #                 sync_mode.tick_no_data()
    #                 wait_frame_ticks += 1
    #
    #             _, rgb_data, depth_data, semantic_data = sync_mode.tick(timeout=2.0)
    #
    #             # Processing raw data
    #             rgb_array = self.process_rgb_img(rgb_data, sensor_width, sensor_height)
    #             depth_array = self.process_depth_data(depth_data, sensor_width, sensor_height)
    #             semantic_array = self.process_semantic_img(semantic_data, sensor_width, sensor_height)
    #
    #             cv2.imshow('rgb', rgb_array)
    #             cv2.waitKey(10)
    #
    #             ego_speed = ego_vehicle.get_velocity()
    #             ego_speed = np.array([ego_speed.x, ego_speed.y, ego_speed.z])
    #             timestamp = round(time.time() * 1000.0)
    #
    #             # Saving into opened HDF5 dataset file
    #             self.HDF5_file.record_data(rgb_array, depth_array, semantic_array, ego_speed, timestamp)
    #             current_ego_recorded_frames += 1
    #             self.total_recorded_frames += 1
    #             timestamps.append(timestamp)
    #
    #             sys.stdout.write("\r")
    #             sys.stdout.write('Frame {0}/{1}'.format(
    #                 self.total_recorded_frames, frames_to_record_one_ego * egos_to_run * len(self.weather_options)))
    #             sys.stdout.flush()
