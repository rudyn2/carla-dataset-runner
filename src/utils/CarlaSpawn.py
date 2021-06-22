import carla
import random
import logging


class CarlaSpawn(object):
    def __init__(self,
                 carla_client,
                 delta_seconds: float,
                 tm_port: int = 8000,
                 distance_to_leading_vehicle: float = 1.0):
        """
        Spawn manager. It is used to spawn actors and track their representations in it's local state.
        It is designed only for simulations on synchronous mode.
        """
        self.client = carla_client

        self.vehicles_list = []
        self.walkers_list = []
        self.all_id = []
        self.all_actors = []

        # get some important objects from the world
        self.world = carla_client.get_world()
        self.map = self.world.get_map()
        self.delta_seconds = delta_seconds
        self.blueprints_vehicles = self.world.get_blueprint_library().filter("vehicle.*")
        self.blueprints_walkers = self.world.get_blueprint_library().filter("walker.pedestrian.*")

        settings = self.world.get_settings()
        self.synchronous_master = settings.synchronous_mode
        if not settings.synchronous_mode:
            self.synchronous_master = True
            settings.synchronous_mode = True
            settings.fixed_delta_seconds = delta_seconds
            self.world.apply_settings(settings)

    def spawn_actors(self,
                     vehicles: int,
                     walkers: int,
                     percentage_pedestrians_running: float = 0.1,
                     percentage_pedestrians_crossing: float = 0.2):
        """
        Spawn a certain amount of vehicles and walkers.
        """

        # get blueprints of vehicles that aren't prone to collide
        bps = [x for x in self.blueprints_vehicles if int(x.get_attribute('number_of_wheels')) == 4]
        bps = [x for x in bps if not x.id.endswith('isetta')]
        bps = [x for x in bps if not x.id.endswith('carlacola')]
        bps = [x for x in bps if not x.id.endswith('cybertruck')]
        bps = [x for x in bps if not x.id.endswith('t2')]
        bps = sorted(bps, key=lambda bp: bp.id)

        spawn_points = self.map.get_spawn_points()
        number_of_spawn_points = len(spawn_points)

        if vehicles < number_of_spawn_points:
            random.shuffle(spawn_points)
        elif vehicles > number_of_spawn_points:
            msg = 'requested %d vehicles, but could only find %d spawn points'
            logging.warning(msg, vehicles, number_of_spawn_points)
            vehicles = number_of_spawn_points

        spawn_actor = carla.command.SpawnActor
        set_autopilot = carla.command.SetAutopilot
        future_actor = carla.command.FutureActor

        # region: SPAWN VEHICLES
        batch = []
        for n, transform in enumerate(spawn_points):
            if n >= vehicles:
                break
            blueprint = random.choice(bps)
            if blueprint.has_attribute('color'):
                color = random.choice(blueprint.get_attribute('color').recommended_values)
                blueprint.set_attribute('color', color)
            if blueprint.has_attribute('driver_id'):
                driver_id = random.choice(blueprint.get_attribute('driver_id').recommended_values)
                blueprint.set_attribute('driver_id', driver_id)
            blueprint.set_attribute('role_name', 'autopilot')

            # spawn the cars and set their autopilot and light state all together
            batch.append(spawn_actor(blueprint, transform)
                         .then(set_autopilot(future_actor, True))
                         )

        for response in self.client.apply_batch_sync(batch, self.synchronous_master):
            if response.error:
                logging.error(response.error)
            else:
                self.vehicles_list.append(response.actor_id)
        # endregion

        # region: SPAWN WALKERS
        # 1. take all the random locations to spawn
        spawn_points = []
        for i in range(walkers):
            spawn_point = carla.Transform()
            loc = self.world.get_random_location_from_navigation()
            if loc is not None:
                spawn_point.location = loc
                spawn_points.append(spawn_point)
        # 2. we spawn the walker object
        batch = []
        for spawn_point in spawn_points:
            walker_bp = random.choice(self.blueprints_walkers)
            # set as not invencible
            if walker_bp.has_attribute('is_invincible'):
                walker_bp.set_attribute('is_invincible', 'false')
            batch.append(spawn_actor(walker_bp, spawn_point))

        results = self.client.apply_batch_sync(batch, True)

        for i in range(len(results)):
            if results[i].error:
                logging.error(results[i].error)
            else:
                self.walkers_list.append({"id": results[i].actor_id})
        # 3. we spawn the walker controller
        batch = []
        walker_controller_bp = self.world.get_blueprint_library().find('controller.ai.walker')
        for i in range(len(self.walkers_list)):
            batch.append(spawn_actor(walker_controller_bp, carla.Transform(), self.walkers_list[i]["id"]))
        results = self.client.apply_batch_sync(batch, True)
        for i in range(len(results)):
            if results[i].error:
                logging.error(results[i].error)
            else:
                self.walkers_list[i]["con"] = results[i].actor_id
        # 4. we put altogether the walkers and controllers id to get the objects from their id
        for i in range(len(self.walkers_list)):
            self.all_id.append(self.walkers_list[i]["con"])
            self.all_id.append(self.walkers_list[i]["id"])
        all_actors = self.world.get_actors(self.all_id)

        # wait for a tick to ensure client receives the last transform of the walkers we have just created
        self.world.tick()

        # 5. initialize each controller and set target to walk to (list is [controler, actor, controller, actor ...])
        for i in range(0, len(self.all_id), 2):
            # start walker
            all_actors[i].start()
            # set walk to random point
            all_actors[i].go_to_location(self.world.get_random_location_from_navigation())
            # random max speed
            all_actors[i].set_max_speed(1 + random.random())  # max speed between 1 and 2 (default is 1.4 m/s)

        print('spawned %d vehicles and %d walkers' % (len(self.vehicles_list), len(self.walkers_list)))

    def destroy_actors(self):
        """
        Destroy all the actors tracked in the local state.
        """
        # stop synchronous mode before destroying actors
        settings = self.world.get_settings()
        settings.synchronous_mode = False
        settings.fixed_delta_seconds = None
        self.world.apply_settings(settings)

        self.client.apply_batch([carla.command.DestroyActor(x) for x in self.vehicles_list])

        # stop walker controllers (list is [controller, actor, controller, actor ...])
        for i in range(0, len(self.all_id), 2):
            self.all_actors[i].stop()

        self.client.apply_batch([carla.command.DestroyActor(x) for x in self.all_id])

        self.vehicles_list = []
        self.walkers_list = []
        self.all_id = []
        self.all_actors = []
