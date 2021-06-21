from custom_agents.noisy_agent import NoisyAgent
import carla
from termcolor import colored
from utils.SensorHandlers import on_collision
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
        print(colored(f"Successfully connected to CARLA at {host}:{port}", "green"))

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

    def set_spectator(self, vehicle):
        """
        The following code would move the spectator actor, to point the view towards a desired vehicle.
        """
        spectator = self.world.get_spectator()
        transform = vehicle.get_transform()
        spectator.set_transform(carla.Transform(transform.location + carla.Location(z=80),
                                                carla.Rotation(pitch=-90)))

    def set_sensors(self, vehicle):
        print(colored("[*] Setting sensors", "white"))
        rgb_camera = self.set_camera(vehicle)
        depth_sensor = self.set_depth_sensor(vehicle)
        semantic_sensor = self.set_semantic_sensor(vehicle)
        collision_sensor = self.set_collision_sensor(vehicle)
        print(colored("[+] All sensors were attached successfully", "green"))

    def set_ego(self, noisy: bool = True):
        """
        Add ego agent to the simulation. Return an Carla.Vehicle object.
        :return: The ego agent and ego vehicle if it was added successfully. Otherwise returns None.
        """
        # These vehicles are not considered because
        # the cameras get occluded without changing their absolute position
        available_vehicle_bps = [bp for bp in self.blueprint_library.filter("vehicle.*")]
        ego_vehicle_bp = random.choice([x for x in available_vehicle_bps if x.id not in
                                     ['vehicle.audi.tt', 'vehicle.carlamotors.carlacola', 'vehicle.tesla.cybertruck',
                                      'vehicle.volkswagen.t2']])

        spawn_points = self.map.get_spawn_points()
        random.shuffle(spawn_points)

        ego_vehicle = self.try_spawn_ego(ego_vehicle_bp, spawn_points)
        if ego_vehicle is None:
            print(colored("Couldn't spawn ego vehicle", "red"))
            return None
        self.set_spectator(ego_vehicle)

        ego_agent = NoisyAgent(ego_vehicle, is_noisy=noisy)
        ego_vehicle_location = ego_vehicle.get_location()

        for v in self.world.get_actors().filter("vehicle.*"):
            if v.id != ego_vehicle.id:
                v.set_autopilot(True)

        # find the first location that isn't the ego vehicle location and its far away
        ego_setup_success = False
        for point in spawn_points:
            dx = point.location.x - ego_vehicle_location.x
            dy = point.location.y - ego_vehicle_location.y
            distance = np.sqrt(dx * dx + dy * dy)
            if point.location != ego_vehicle.get_location() and distance > 20:

                ego_agent.set_route(ego_vehicle.get_location(), point.location)
                break

        return ego_agent, ego_vehicle

    def try_spawn_ego(self, ego_vehicle_bp, spawn_points):
        ego_vehicle = None
        for p in spawn_points:
            ego_vehicle = self.world.try_spawn_actor(ego_vehicle_bp, p)
            if ego_vehicle:
                ego_vehicle.set_autopilot(False)
                return ego_vehicle
        return ego_vehicle

    def record(self):
        """
        Use the ego to record all the data in one episode. Returns the data needed for hdf5 saver and json saver.
        # TODO: Specify method signature.
        """
        raise NotImplementedError


if __name__ == '__main__':

    c = CarlaExtractor()
    _, v = c.set_ego()
    c.set_sensors(v)
