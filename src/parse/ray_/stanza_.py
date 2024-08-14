import ray

from parse.stanza_ import StanzaRunner


@ray.remote(num_cpus=1, num_gpus=1, max_restarts=3, max_task_retries=-1)
class StanzaActor(StanzaRunner):
    pass
