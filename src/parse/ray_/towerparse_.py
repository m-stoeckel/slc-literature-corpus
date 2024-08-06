import ray

from parse.towerparse_ import TowerParseRunner


@ray.remote(num_cpus=1, num_gpus=1, max_restarts=3, max_task_retries=-1)
class TowerParseActor(TowerParseRunner):
    pass
