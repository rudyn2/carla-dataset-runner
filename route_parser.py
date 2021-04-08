import os
import xml.etree.ElementTree as ET

import carla


class RouteParser:

    def __init__(self, map_, path: str):
        self.path = path
        self.map = map_

    def parse_file(self):
        if not os.path.exists(self.path):
            raise ValueError("Provided path doesn't exist in file directory")

        root = ET.parse(self.path).getroot()
        parsed_routes = {}
        for route in root:
            parsed_routes[route.attrib['id']] = {'town': route.attrib['town'], 'waypoints': []}
            for wp in route:
                if wp.tag == 'waypoint':
                    parsed_routes[route.attrib['id']]['waypoints'].append(self.parse_locs(wp.attrib))
        return parsed_routes

    def parse_locs(self, wp_as_dict):
        rotation = carla.Rotation(pitch=float(wp_as_dict['pitch']), yaw=float(wp_as_dict['yaw']),
                                  roll=float(wp_as_dict['roll']))
        location = carla.Location(x=float(wp_as_dict['x']), y=float(wp_as_dict['y']), z=float(wp_as_dict['z']))
        return location
