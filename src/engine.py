class CarlaExtractor(object):
    """
    Class created to orchestrate the simulation runtime. It should be able to spawn vehicles, spawn pedestrians,
    set routes, set sensors, etc.
    """

    def __init__(self):
        pass

    def reset(self):
        raise NotImplementedError

    def set_actors(self, vehicles: int, walkers: int):
        raise NotImplementedError

    def set_sensors(self):
        raise NotImplementedError

    def set_ego(self):
        """
        Add ego agent to the simulation. Return an Carla.Vehicle object.
        :return: Carla.Vehicle
        """
        raise NotImplementedError

    def record(self):
        """
        Use the ego to record all the data in one episode. Returns the data needed for hdf5 saver and json saver.
        # TODO: Specify method signature.
        """
        raise NotImplementedError
