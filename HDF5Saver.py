import h5py
import numpy as np


class HDF5Saver:
    def __init__(self, sensor_width, sensor_height, file_path_to_save="data/carla_dataset.hdf5"):
        self.sensor_width = sensor_width
        self.sensor_height = sensor_height

        self.file = h5py.File(file_path_to_save, "w")
        # Creating groups to store each type of data
        self.rgb_group = self.file.create_group("rgb")
        self.depth_group = self.file.create_group("depth")
        self.semantic_group = self.file.create_group("semantic")
        self.ego_speed_group = self.file.create_group("ego_speed")
        self.timestamp_group = self.file.create_group("timestamps")

        # Storing metadata
        self.file.attrs['sensor_width'] = sensor_width
        self.file.attrs['sensor_height'] = sensor_height
        self.file.attrs['simulation_synchronization_type'] = "syncd"
        self.rgb_group.attrs['channels'] = 'R,G,B'
        self.ego_speed_group.attrs['x,y,z_velocity'] = 'in m/s'
        self.timestamp_group.attrs['time_format'] = "current time in MILISSECONDS since the unix epoch " \
                                                    "(time.time()*1000 in python3)"

    def record_data(self, rgb_array, depth_array, semantic_array, ego_speed, timestamp):
        timestamp = str(timestamp)
        self.rgb_group.create_dataset(timestamp, data=rgb_array)
        self.depth_group.create_dataset(timestamp, data=depth_array)
        self.semantic_group.create_dataset(timestamp, data=semantic_array)
        self.ego_speed_group.create_dataset(timestamp, data=ego_speed)

    def record_all_timestamps(self, timestamps_list):
        self.timestamp_group.create_dataset("timestamps", data=np.array(timestamps_list))

    def close_HDF5(self):
        self.file.close()
