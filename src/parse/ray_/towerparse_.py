import ray

from parse.ray_.stanza_ import ray_remote_wrapper
from parse.towerparse_ import TowerParser


@ray_remote_wrapper(
    num_cpus=1,
    num_gpus=1,
    max_restarts=-1,
    max_task_retries=3,
)
class TowerParseActor(TowerParser):
    pass
