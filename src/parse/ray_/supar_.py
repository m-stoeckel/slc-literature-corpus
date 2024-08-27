import ray

from parse.ray_.stanza_ import ray_remote_wrapper
from parse.supar_ import SuparParser


@ray_remote_wrapper(
    num_cpus=1,
    num_gpus=1,
    max_restarts=-1,
    max_task_retries=3,
)
class SuparActor(SuparParser):
    def __init__(self, *args, **kwargs):
        kwargs.setdefault("quiet", True)
        super().__init__(*args, **kwargs)
