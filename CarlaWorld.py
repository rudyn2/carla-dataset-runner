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
from agents.navigation.global_route_planner import GlobalRoutePlanner
from agents.navigation.global_route_planner_dao import GlobalRoutePlannerDAO


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


class CantSpawnEgoError(Exception):
    pass


class CarlaWorld:
    def __init__(self, host: str, port: int, town='Town02'):
        self.world_tag = town

        # Carla initialization
        self.host = host
        self.port = port
        self.client = carla.Client(self.host, self.port)
        self.client.set_timeout(20.0)
        self.client.load_world(town)
        self.world = self.client.get_world()

        print(f'Successfully connected to CARLA {self.host}:{self.port}')
        self.blueprint_library = self.world.get_blueprint_library()
        self.map = self.world.get_map()
        self.ego_vehicle = None
        self.ego_agent = None

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
        self.NPC = NPCClass(self.host)
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

    @staticmethod
    def compute_distance(loc1, loc2):
        dx = loc1.x - loc2.x
        dy = loc1.y - loc2.y
        return np.sqrt(dx * dx + dy * dy)

    def remove_sensors(self):
        for sensor in self.sensors_list:
            sensor.destroy()
        self.sensors_list = []

    def set_ego_agent(self, route=None, debug: bool = False):

        if route is None:

            # These vehicles are not considered because
            # the cameras get occluded without changing their absolute position
            ego_vehicle = random.choice([x for x in self.world.get_actors().filter("vehicle.*") if x.type_id not in
                                         ['vehicle.audi.tt', 'vehicle.carlamotors.carlacola', 'vehicle.volkswagen.t2']])
            ego_agent = BehaviorAgent(ego_vehicle)

            spawn_points = self.map.get_spawn_points()
            random.shuffle(spawn_points)
            if spawn_points[0].location != ego_agent.vehicle.get_location():
                destination = spawn_points[0].location
            else:
                destination = spawn_points[1].location
            ego_agent.set_destination(ego_agent.vehicle.get_location(), destination, clean=True)

        else:
            vehicle_bp = self.blueprint_library.find('vehicle.tesla.model3')
            dense_route = self.interpolate_trajectory(route, hop_resolution=1.0)
            if debug:
                # 2.2 Draw waypoints
                for i, t in enumerate(dense_route):
                    self.world.debug.draw_string(t[0].transform.location, str(i), draw_shadow=False,
                                                 color=carla.Color(r=255, g=0, b=0), life_time=30,
                                                 persistent_lines=True)

            not_spawned = True
            route_p_spawn = 0
            while not_spawned:
                try:
                    start_wp = self.map.get_waypoint(dense_route[route_p_spawn][0].transform.location)
                    point_to_spawn = self.find_closest_point_to_spawn(start_wp)
                    ego_vehicle = self.world.spawn_actor(vehicle_bp, point_to_spawn)
                    not_spawned = False
                except RuntimeError:
                    route_p_spawn += 1
                except IndexError:
                    raise CantSpawnEgoError("Couldn't spawn ego for provided route")

            ego_agent = BehaviorAgent(ego_vehicle)
            ego_agent.set_predefined_route(dense_route)

        ego_agent.update_information()
        print("Behavior Agent has been set up successfully")
        self.ego_vehicle = ego_vehicle
        self.ego_agent = ego_agent

    def find_closest_point_to_spawn(self, wp):

        closest_points = []
        for p in self.map.get_spawn_points():
            distance = self.compute_distance(wp.transform.location, p.location)
            if distance < 10:
                closest_points.append((p, distance))

        sorted_closest_points = sorted(closest_points, key=lambda p: p[1])
        closest_points = [p[0] for p in sorted_closest_points]

        closest_valid_point = None
        for p in closest_points:
            wp_p = self.map.get_waypoint(p.location)
            if wp_p.lane_id == wp.lane_id:
                closest_valid_point = p

        if closest_valid_point is None:
            raise RuntimeError("Can't find valid point to spawn ego vehicle")

        return closest_valid_point

    def interpolate_trajectory(self, waypoints_trajectory, hop_resolution=1.0):
        """
        Given some raw keypoints interpolate a full dense trajectory to be used by the user.
        returns the full interpolated route both in GPS coordinates and also in its original form.

        Args:
            - world: an reference to the CARLA world so we can use the planner
            - waypoints_trajectory: the current coarse trajectory
            - hop_resolution: is the resolution, how dense is the provided trajectory going to be made
        """

        dao = GlobalRoutePlannerDAO(self.world.get_map(), hop_resolution)
        grp = GlobalRoutePlanner(dao)
        grp.setup()
        # Obtain route plan
        route = []
        for i in range(len(waypoints_trajectory) - 1):  # Goes until the one before the last.

            waypoint = waypoints_trajectory[i]
            waypoint_next = waypoints_trajectory[i + 1]
            interpolated_trace = grp.trace_route(waypoint, waypoint_next)

            for wp_tuple in interpolated_trace:
                route.append((wp_tuple[0], wp_tuple[1]))

        return route

    def begin_data_acquisition(self, sensor_width, sensor_height, fov, frames_to_record_one_ego=300,
                               route=None, debug: bool = False):
        """
        Records data using one ego for 'frames_to_record_one_ego' frames.
        """

        print("Beginning new data acquisition")
        # Changes the ego vehicle to be put the sensor
        current_ego_recorded_frames = 0
        media_data = []
        info_data = []

        ego_vehicle = self.ego_vehicle
        ego_agent = self.ego_agent
        ego_reached_destination = False

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
                if current_ego_recorded_frames == frames_to_record_one_ego or ego_reached_destination:
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

                if self.compute_distance(ego_vehicle.get_transform().location, route[-1]) < 10:
                    print("Ego has reached destination successfully")
                    ego_reached_destination = True

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
