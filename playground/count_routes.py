import argparse
import os
import xml.etree.ElementTree as ET
from collections import defaultdict


def main(path: str):
    if not os.path.exists(path):
        raise ValueError("Provided path doesn't exist in file directory")

    root = ET.parse(path).getroot()
    route_count = defaultdict(int)
    for route in root:
        route_count[route.attrib['town']] += 1
    for town in route_count.keys():
        print(f"{town}: {route_count[town]}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Count routes per town in xml file",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-i', '--input', required=True, default=None, type=str, help='path to xml file')
    args = parser.parse_args()
    main(args.input)
