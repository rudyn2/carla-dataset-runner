from custom_agents.noisy_agent import NoisyAgent
import carla
from termcolor import colored
from utils.SensorHanlers import on_collision
import numpy as np
import weakref
import random


class CarlaExtractor(object):
    """
    Class created to orchestrate the simulation runtime. It should be able to spawn vehicles, spawn pedestrians,
    set routes, set sensors, etc.
    """

    def __init__(self,
                 host: str = "localhost",
                 port: int = 2000,
                 town: str = 'Town01'):

        print(colored("Connecting to CARLA...", "white"))
        self.client = carla.Client(host, port)
        self.client.set_timeout(10.0)
        self.client.load_world(town)
        self.world = self.client.get_world()
        self.blueprint_library = self.world.get_blueprint_library()
        self.map = self.world.get_map()
        print(f"Successfully connected to CARLA at {host}:{port}")

        self.sensor_list = []

    def reset(self):
        # TODO: Figure out how to reset the simulation cleanly
        raise NotImplementedError

    def set_actors(self, vehicles: int, walkers: int):
        raise NotImplementedError

    def set_camera(self, vehicle, sensor_width: int = 640, sensor_height: int = 480, fov: int = 110) -> object:
        bp = self.blueprint_library.find('sensor.camera.rgb')
        bp.set_attribute('image_size_x', f'{sensor_width}')
        bp.set_attribute('image_size_y', f'{sensor_height}')
        bp.set_attribute('fov', f'{fov}')

        # Adjust sensor relative position to the vehicle
        spawn_point = carla.Transform(carla.Location(x=1.0, z=2.0))
        rgb_camera = self.world.spawn_actor(bp, spawn_point, attach_to=vehicle)
        rgb_camera.blur_amount = 0.0
        rgb_camera.motion_blur_intensity = 0
        rgb_camera.motion_max_distortion = 0

        # Camera calibration
        calibration = np.identity(3)
        calibration[0, 2] = sensor_width / 2.0
        calibration[1, 2] = sensor_height / 2.0
        calibration[0, 0] = calibration[1, 1] = sensor_width / (2.0 * np.tan(fov * np.pi / 360.0))
        return rgb_camera

    def set_depth_sensor(self, vehicle, sensor_width: int = 640, sensor_height: int = 480, fov: int = 110):
        bp = self.blueprint_library.find('sensor.camera.depth')
        bp.set_attribute('image_size_x', f'{sensor_width}')
        bp.set_attribute('image_size_y', f'{sensor_height}')
        bp.set_attribute('fov', f'{fov}')

        # Adjust sensor relative position to the vehicle
        spawn_point = carla.Transform(carla.Location(x=1.0, z=2.0))
        depth_camera = self.world.spawn_actor(bp, spawn_point, attach_to=vehicle)
        return depth_camera

    def set_semantic_sensor(self, vehicle, sensor_width: int = 640, sensor_height: int = 480, fov: int = 110):
        bp = self.blueprint_library.find('sensor.camera.semantic_segmentation')
        bp.set_attribute('image_size_x', f'{sensor_width}')
        bp.set_attribute('image_size_y', f'{sensor_height}')
        bp.set_attribute('fov', f'{fov}')

        # Adjust sensor relative position to the vehicle
        spawn_point = carla.Transform(carla.Location(x=1.0, z=2.0))
        semantic_camera = self.world.spawn_actor(bp, spawn_point, attach_to=vehicle)
        return semantic_camera

    def set_collision_sensor(self, vehicle):
        """
        In case of collision, this sensor will update the 'collision_info' attribute with a dictionary that contains
        the following keys: ["frame", "actor_id", "other_actor"].
        """
        bp = self.blueprint_library.find('sensor.other.collision')
        collision_sensor = self.world.spawn_actor(bp, carla.Transform(), attach_to=vehicle)
        weak_self = weakref.ref(self)
        collision_sensor.listen(lambda event: on_collision(weak_self, event))
        return collision_sensor

    def set_birdeye_camera(self, vehicle, sensor_width: int = 640, sensor_height: int = 480, fov: int = 110):
        bp = self.blueprint_library.find('sensor.camera.rgb')
        bp.set_attribute('image_size_x', f'{sensor_width}')
        bp.set_attribute('image_size_y', f'{sensor_height}')
        bp.set_attribute('fov', f'{fov}')

        # Adjust sensor relative position to the vehicle
        spawn_point = carla.Transform(carla.Location(x=1, z=20), carla.Rotation(pitch=-90, yaw=0, roll=0))
        birdeye_camera = self.world.spawn_actor(bp, spawn_point, attach_to=vehicle)
        birdeye_camera.blur_amount = 0.0
        birdeye_camera.motion_blur_intensity = 0
        birdeye_camera.motion_max_distortion = 0

        # Camera calibration
        calibration = np.identity(3)
        calibration[0, 2] = sensor_width / 2.0
        calibration[1, 2] = sensor_height / 2.0
        calibration[0, 0] = calibration[1, 1] = sensor_width / (2.0 * np.tan(fov * np.pi / 360.0))
        birdeye_camera.calibration = calibration  # Parameter K of the camera
        return birdeye_camera

    def set_sensors(self, vehicle, add_birdeye: bool = False):
        print(colored("[*] Setting sensors", "white"))
        rgb_camera = self.set_camera(vehicle)
        depth_sensor = self.set_depth_sensor(vehicle)
        semantic_sensor = self.set_semantic_sensor(vehicle)
        collision_sensor = self.set_collision_sensor(vehicle)
        if add_birdeye:
            birdeye = self.set_birdeye_camera(vehicle)
        print(colored("[+] All sensors were attached successfully", "green"))

    def set_ego(self, noisy: bool = True) -> bool:
        """
        Add ego agent to the simulation. Return an Carla.Vehicle object.
        :return: True if the ego agent was added successfully to the simulation.
        """
        # These vehicles are not considered because
        # the cameras get occluded without changing their absolute position
        ego_vehicle = random.choice([x for x in self.world.get_actors().filter("vehicle.*") if x.type_id not in
                                     ['vehicle.audi.tt', 'vehicle.carlamotors.carlacola', 'vehicle.tesla.cybertruck',
                                      'vehicle.volkswagen.t2']])
        ego_vehicle.set_autopilot(False)
        ego_agent = NoisyAgent(ego_vehicle, is_noisy=noisy)
        ego_vehicle_location = ego_vehicle.get_location()

        for v in self.world.get_actors().filter("vehicle.*"):
            if v.id != ego_vehicle.id:
                v.set_autopilot(True)

        spawn_points = self.map.get_spawn_points()
        random.shuffle(spawn_points)
        # find the first location that isn't the ego vehicle location and its far away
        ego_setup_success = False
        for point in spawn_points:
            dx = point.location.x - ego_vehicle_location.location.x
            dy = point.location.y - ego_vehicle_location.y
            distance = np.sqrt(dx * dx + dy * dy)
            if point.location != ego_vehicle.get_location() and distance > 20:
                ego_setup_success = True
                ego_agent.set_route(ego_vehicle.get_location(), point)
                break

        return ego_setup_success

    def record(self):
        """
        Use the ego to record all the data in one episode. Returns the data needed for hdf5 saver and json saver.
        # TODO: Specify method signature.
        """
        raise NotImplementedError


if __name__ == '__main__':
    c = CarlaExtractor()
