import ray

from parse.ray_.mixin import ray_remote_wrapper
from parse.spacy_ import SpacyParser


@ray_remote_wrapper(
    num_cpus=1,
    num_gpus=0.2,
    max_restarts=-1,
    max_task_retries=3,
    scheduling_strategy="SPREAD",
)
class SpacyActor(SpacyParser):
    def __init__(self, *args, **kwargs):
        kwargs.setdefault("quiet", True)
        super().__init__(*args, **kwargs)
