import sys
import pathlib

# Add carla egg
if sys.platform.__contains__('win'):
    sys.path.append(str(pathlib.Path('../carla_egg/carla-0.9.11-py3.7-win-amd64.egg').absolute()))
else:
    sys.path.append(str(pathlib.Path('../carla_egg/carla-0.9.11-py3.7-linux-x86_64.egg').absolute()))

# Add PythonAPI
sys.path.append(str(pathlib.Path('../PythonAPI').absolute()))
