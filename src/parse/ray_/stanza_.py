import ray

from parse.ray_.mixin import ray_remote_wrapper
from parse.stanza_ import StanzaParser


@ray_remote_wrapper(
    num_cpus=1,
    num_gpus=1,
    max_restarts=-1,
    max_task_retries=3,
)
class StanzaActor(StanzaParser):
    pass
