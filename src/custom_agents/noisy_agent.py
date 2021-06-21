import carla
import numpy as np
from agents.tools.vehicle_position import get_vehicle_position, get_vehicle_orientation
from controller import PIDController

from roaming import RoamingAgentMine


class NoisyAgent(RoamingAgentMine):
    """
    Each parameter is in units of frames.
    State can be "drive" or "noise".
    """

    def __init__(self, vehicle, noise=None, is_noisy=False):
        super().__init__(vehicle, resolution=1, threshold_before=7.5, threshold_after=5.)

        if is_noisy:
            self.params = {'drive': (100, 'noise'), 'noise': (10, 'drive')}
        else:
            self.params = {'drive': (100, 'drive')}

        self.steps = 0
        self.state = 'drive'
        self.noise_steer = 0
        self.last_throttle = 0
        self.noise_func = noise if noise else lambda: np.random.uniform(-0.25, 0.25)

        self.speed_control = PIDController(K_P=0.5, K_I=0.5 / 20, K_D=0.1)
        self.turn_control = PIDController(K_P=0.75, K_I=1.0 / 20, K_D=0.0)

    def run_step(self, debug=False):
        self.steps += 1

        last_status = self.state
        num_steps, next_state = self.params[self.state]
        real_control, blocking_light, traffic_light = super().run_step(debug)
        real_control.throttle *= max((1.0 - abs(real_control.steer)), 0.25)

        control = carla.VehicleControl()
        control.manual_gear_shift = False

        if self.state == 'noise':
            control.steer = self.noise_steer
            control.throttle = self.last_throttle
        else:
            control.steer = real_control.steer
            control.throttle = real_control.throttle
            control.brake = real_control.brake

        if self.steps == num_steps:
            self.steps = 0
            self.state = next_state
            self.noise_steer = self.noise_func()
            self.last_throttle = control.throttle

        self.debug = {
            'waypoint': (self.waypoint.x, self.waypoint.y, self.waypoint.z),
            'vehicle': (self.vehicle.x, self.vehicle.y, self.vehicle.z)
        }

        #####################
        speed = self._vehicle.get_velocity()
        speed_x, speed_y, speed_z = speed.x, speed.y, speed.z
        speed = np.linalg.norm([speed.x, speed.y])
        speed_limit = self._vehicle.get_speed_limit()

        tl_state = "Red" if blocking_light else "Green"
        tl_distance = None
        if traffic_light:
            dx = traffic_light.get_location().x - self._vehicle.get_location().x
            dy = traffic_light.get_location().y - self._vehicle.get_location().y
            tl_distance = np.sqrt(dx * dx + dy * dy)

        position = get_vehicle_position(self._map, self._vehicle)
        orientation = get_vehicle_orientation(self._map, self._vehicle)

        #####################
        info_dict = {
            "control": control,
            "command": self.road_option.name,
            "speed_x": speed_x,
            "speed_y": speed_y,
            "speed_z": speed_z,
            "speed": speed,
            "speed_limit": speed_limit,
            "at_tl": blocking_light,
            "tl_state": tl_state,
            "tl_distance": tl_distance,
            "lane_distance": position,
            "lane_orientation": orientation,
            "collision": None
        }
        return info_dict
