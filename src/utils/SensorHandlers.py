import numpy as np


def process_depth_data(data, sensor_width, sensor_height):
    """
    normalized = (R + G * 256 + B * 256 * 256) / (256 * 256 * 256 - 1)
    in_meters = 1000 * normalized
    """
    data = np.array(data.raw_data)
    data = data.reshape((sensor_height, sensor_width, 4))
    data = data.astype(np.float32)
    # Apply (R + G * 256 + B * 256 * 256) / (256 * 256 * 256 - 1).
    normalized_depth = np.dot(data[:, :, :3], [65536.0, 256.0, 1.0])
    normalized_depth /= 16777215.0  # (256.0 * 256.0 * 256.0 - 1.0)
    depth_meters = normalized_depth * 1000
    return depth_meters


def process_rgb_img(img, sensor_width, sensor_height):
    img = np.array(img.raw_data)
    img = img.reshape((sensor_height, sensor_width, 4))
    img = img[:, :, :3]  # taking out opacity channel
    return img


def process_semantic_img(img, sensor_width, sensor_height):
    img = np.array(img.raw_data)
    img = img.reshape((sensor_height, sensor_width, 4))
    img = img[:, :, 2]  # taking just the RED channel
    return img


def on_collision(weak_self, event, ego_id):
    """On collision method"""
    self = weak_self()
    if not self:
        return

    if isinstance(event, list):
        for e in event:
            if e.actor.id == ego_id:
                self.collision_info = {
                    "frame": e.frame,
                    "actor_id": e.actor.id,
                    "other_actor": e.other_actor.type_id,
                }
                return
        return

    else:   # we assume that there is just one collision event
        if event.actor.id == ego_id:
            self.collision_info = {
                "frame": event.frame,
                "actor_id": event.actor.id,
                "other_actor": event.other_actor.type_id,
            }

