from .custom_agents.noisy_agent import NoisyAgent
import carla
from termcolor import colored
from .utils.SensorHandlers import on_collision, process_rgb_img, process_depth_data, process_semantic_img
from .utils.SyncMode import CarlaSyncMode
from .utils.CarlaSpawn import CarlaSpawn
import numpy as np
import weakref
import random
import time
from tqdm import tqdm


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


class CarlaExtractor(object):
    """
    Class created to orchestrate the simulation runtime. It should be able to spawn vehicles, spawn pedestrians,
    set routes, set sensors, etc.
    """

    def __init__(self,
                 sensor_width: int = 288,
                 sensor_height: int = 288,
                 host: str = "localhost",
                 port: int = 2000,
                 town: str = 'Town01',
                 fps: int = 30):

        print(colored("Connecting to CARLA...", "white"))
        self.client = carla.Client(host, port)
        self.client.set_timeout(20.0)
        self.client.load_world(town)
        self.world = self.client.get_world()
        self.blueprint_library = self.world.get_blueprint_library()
        self.map = self.world.get_map()
        self.spawn_manager = CarlaSpawn(carla_client=self.client, delta_seconds=1/fps)
        print(colored(f"Successfully connected to CARLA at {host}:{port}", "green"))

        self.sensor_width, self.sensor_height = sensor_width, sensor_height
        self.fps = fps
        self.sensor_list = []
        self.collision_info = {}

    def set_weather(self, weather_option):
        weather = carla.WeatherParameters(*weather_option)
        self.world.set_weather(weather)

    def set_actors(self, vehicles: int, walkers: int):
        self.spawn_manager.spawn_actors(vehicles, walkers)

    def set_camera(self, vehicle, sensor_width: int, sensor_height: int, fov: int) -> object:
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

    def set_depth_sensor(self, vehicle, sensor_width: int, sensor_height: int, fov: int):
        bp = self.blueprint_library.find('sensor.camera.depth')
        bp.set_attribute('image_size_x', f'{sensor_width}')
        bp.set_attribute('image_size_y', f'{sensor_height}')
        bp.set_attribute('fov', f'{fov}')

        # Adjust sensor relative position to the vehicle
        spawn_point = carla.Transform(carla.Location(x=1.0, z=2.0))
        depth_camera = self.world.spawn_actor(bp, spawn_point, attach_to=vehicle)
        return depth_camera

    def set_semantic_sensor(self, vehicle, sensor_width: int, sensor_height: int, fov: int):
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
        collision_sensor.listen(lambda event: on_collision(weak_self, event, vehicle.id))
        return collision_sensor

    def update_spectator(self, vehicle):
        """
        The following code would move the spectator actor, to point the view towards a desired vehicle.
        """
        spectator = self.world.get_spectator()
        transform = vehicle.get_transform()
        spectator.set_transform(carla.Transform(transform.location + carla.Location(z=50),
                                                carla.Rotation(pitch=-90)))

    def set_sensors(self, vehicle, sensor_width: int, sensor_height: int, fov: int = 110):
        rgb_camera = self.set_camera(vehicle, sensor_width, sensor_height, fov)
        depth_sensor = self.set_depth_sensor(vehicle, sensor_width, sensor_height, fov)
        semantic_sensor = self.set_semantic_sensor(vehicle, sensor_width, sensor_height, fov)
        collision_sensor = self.set_collision_sensor(vehicle)
        return [rgb_camera, depth_sensor, semantic_sensor], collision_sensor

    def set_ego(self, noisy: bool = True):
        """
        Add ego agent to the simulation. Return an Carla.Vehicle object.
        :return: The ego agent and ego vehicle if it was added successfully. Otherwise returns None.
        """
        # These vehicles are not considered because
        # the cameras get occluded without changing their absolute position
        info = {}
        available_vehicle_bps = [bp for bp in self.blueprint_library.filter("vehicle.*")]
        ego_vehicle_bp = random.choice([x for x in available_vehicle_bps if x.id not in
                                     ['vehicle.audi.tt', 'vehicle.carlamotors.carlacola', 'vehicle.tesla.cybertruck',
                                      'vehicle.volkswagen.t2', 'vehicle.bh.crossbike']])

        spawn_points = self.map.get_spawn_points()
        random.shuffle(spawn_points)

        ego_vehicle = self.try_spawn_ego(ego_vehicle_bp, spawn_points)
        if ego_vehicle is None:
            print(colored("Couldn't spawn ego vehicle", "red"))
            return None
        info['vehicle'] = ego_vehicle.type_id
        info['id'] = ego_vehicle.id
        self.update_spectator(vehicle=ego_vehicle)

        ego_agent = NoisyAgent(ego_vehicle, is_noisy=noisy)
        ego_vehicle_location = ego_vehicle.get_location()

        for v in self.world.get_actors().filter("vehicle.*"):
            if v.id != ego_vehicle.id:
                v.set_autopilot(True)

        # find the first location that isn't the ego vehicle location and its far away
        points_distance = []
        for point in spawn_points:
            dx = point.location.x - ego_vehicle_location.x
            dy = point.location.y - ego_vehicle_location.y
            distance = np.sqrt(dx * dx + dy * dy)
            if point.location != ego_vehicle.get_location():
                points_distance.append((point, distance))

        # sort point in descending order according to their distance to the ego
        points_distance = sorted(points_distance, key=lambda x: x[1], reverse=True)
        for point, _ in points_distance:
            try:
                ego_agent.set_route(ego_vehicle.get_location(), point.location)
                info['destination'] = point
                break
            except Exception:
                continue

        return ego_agent, ego_vehicle, info

    def try_spawn_ego(self, ego_vehicle_bp, spawn_points):
        ego_vehicle = None
        for p in spawn_points:
            ego_vehicle = self.world.try_spawn_actor(ego_vehicle_bp, p)
            if ego_vehicle:
                ego_vehicle.set_autopilot(False)
                return ego_vehicle
        return ego_vehicle

    def show_route(self, start_location, end_location):
        self.world.debug.draw_string(start_location, "START", draw_shadow=False,
                                     color=carla.Color(r=255, g=0, b=0), life_time=30, persistent_lines=True)
        self.world.debug.draw_string(end_location, "END", draw_shadow=False,
                                     color=carla.Color(r=255, g=0, b=0), life_time=30, persistent_lines=True)

    def record(self, vehicles: int, walkers: int, noisy: bool = True, max_frames: int = 500, skip_frames: int = 5, debug: bool = False):
        """
        Use the ego to record all the data in one episode. Returns the data needed for hdf5 saver and json saver.
        """

        ego_agent, ego_vehicle, info = self.set_ego(noisy=noisy)
        destination = info['destination']
        # SPAWN SURROUNDING VEHICLES AND PEDESTRIANS HERE
        sensors, collision_sensor = self.set_sensors(ego_vehicle, self.sensor_width, self.sensor_height)
        media, meta = [], []

        try:
            self.set_actors(vehicles=vehicles, walkers=walkers)
            if debug:
                self.show_route(ego_vehicle.get_location(), destination.location)
            desc = "Frame: {}/{} SPEED={:.2f} HLC={} LANE_DISTANCE={:2f} LANE_ORIENTATION={:.2F}"
            pbar = tqdm(initial=0, leave=False, total=max_frames, desc=desc.format(0, max_frames, 0, 0, 0, 0))

            print(colored("[*] Initializing extraction", "white"))
            with CarlaSyncMode(self.world, *sensors, fps=self.fps) as sync_mode:
                # warm-up, put the ego vehicle in movement
                self.skip_frames(ego_agent, ego_vehicle, 15, sync_mode)

                frames = 0
                while frames <= max_frames:

                    self.skip_frames(ego_agent, ego_vehicle, skip_frames, sync_mode)

                    # get data and process it
                    _, rgb, depth, semantic = sync_mode.tick(timeout=2.0)
                    rgb = process_rgb_img(rgb, self.sensor_width, self.sensor_height)
                    depth = process_depth_data(depth, self.sensor_width, self.sensor_height)
                    semantic = process_semantic_img(semantic, self.sensor_width, self.sensor_height)
                    timestamp = round(time.time() * 1000.0)

                    # apply control
                    ego_info = ego_agent.run_step()
                    control = ego_info['control']
                    ego_vehicle.apply_control(control)

                    # save data
                    ego_info["control"] = parse_control(ego_info["control"])
                    media.append(dict(timestamp=timestamp, rgb=rgb, depth=depth, semantic=semantic))
                    meta.append(dict(timestamp=timestamp, metadata=ego_info))

                    # preparing next frame
                    frames += 1
                    pbar.desc = desc.format(frames, max_frames, ego_info["speed"], ego_info["command"],
                                            ego_info["lane_distance"], ego_info["lane_orientation"])
                    pbar.update(1)
                    self.update_spectator(ego_vehicle)

                    # region: check terminal conditions: ego reached destination or an collision has occurred
                    if self._compute_distance(ego_vehicle, destination.location) < 10:
                        tqdm.write(colored("Early stopping due to ego agent reached its destination", "white"))
                        frames = max_frames + 1
                    if self.collision_info:
                        if len(info) > 0:
                            info[-1]['metadata']['collision'] = self.collision_info
                        tqdm.write(colored("Early stopping due to collision", "white"))
                        frames = max_frames + 1
                    # endregion
                pbar.close()
            print(colored("[+] Extraction completed successfully, exiting sync mode...", "green"))
        finally:

            # destroy sensors, ego vehicle and social actors
            for sensor in [*sensors, collision_sensor]:
                if sensor.is_listening:
                    sensor.stop()
                if sensor.is_alive:
                    sensor.destroy()
            if ego_vehicle.is_alive:
                ego_vehicle.destroy()
            self.spawn_manager.destroy_actors()

        return media, meta

    def skip_frames(self, ego_agent, ego_vehicle, skip_frames, sync_mode):
        for _ in range(skip_frames):
            sync_mode.tick_no_data()
            ego_vehicle.apply_control(ego_agent.run_step()['control'])
            self.update_spectator(ego_vehicle)

    @staticmethod
    def _compute_distance(actor, location) -> float:
        actor_loc = actor.get_location()
        dx = actor_loc.x - location.x
        dy = actor_loc.y - location.y
        return np.sqrt(dx * dx + dy * dy)


if __name__ == '__main__':

    c = CarlaExtractor()
    c.record(vehicles=100, walkers=200, max_frames=100, debug=True)
