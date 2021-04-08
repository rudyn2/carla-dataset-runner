import h5py
from tqdm import tqdm


class HDF5Saver:
    def __init__(self, sensor_width, sensor_height, file_path_to_save="data/carla_dataset.hdf5"):
        self.sensor_width = sensor_width
        self.sensor_height = sensor_height

        self.file = h5py.File(file_path_to_save, "w")

        # Creating groups to store each type of data
        self.rgb_group = self.file.create_group("rgb")
        self.depth_group = self.file.create_group("depth")
        self.semantic_group = self.file.create_group("semantic")
        self.timestamp_group = self.file.create_group("timestamps")

        # Storing metadata
        self.file.attrs['sensor_width'] = sensor_width
        self.file.attrs['sensor_height'] = sensor_height
        self.file.attrs['simulation_synchronization_type'] = "syncd"
        self.rgb_group.attrs['channels'] = 'R,G,B'
        self.timestamp_group.attrs['time_format'] = "current time in MILISSECONDS since the unix epoch " \
                                                    "(time.time()*1000 in python3)"

    def save_one_ego_run(self, run_id: str, media_data: list):
        # if a group already exits override its content
        if run_id in self.file.keys():
            del self.file[run_id]

        ego_run_group = self.file.create_group(run_id)
        ego_run_rgb_group = ego_run_group.create_group("rgb")
        ego_run_depth_group = ego_run_group.create_group("depth")
        ego_run_semantic_group = ego_run_group.create_group("semantic")

        for frame_dict in tqdm(media_data, "Saving images "):
            # one frame dict contains rgb, depth and semantic information
            timestamp = str(frame_dict["timestamp"])
            ego_run_rgb_group.create_dataset(timestamp, data=frame_dict["rgb"], compression='gzip')
            ego_run_depth_group.create_dataset(timestamp, data=frame_dict["depth"], compression='gzip')
            ego_run_semantic_group.create_dataset(timestamp, data=frame_dict["semantic"], compression='gzip')

    def close_hdf5(self):
        self.file.close()
