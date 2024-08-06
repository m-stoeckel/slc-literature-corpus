import ray

from parse.spacy_ import SpacyRunner


@ray.remote(
    num_cpus=1,
    num_gpus=0.2,
    max_restarts=3,
    max_task_retries=-1,
    scheduling_strategy="SPREAD",
)
class SpacyActor(SpacyRunner):
    def __init__(
        self,
        *args,
        quiet: bool = True,
        **kwargs,
    ):
        super().__init__(*args, **kwargs, quiet=quiet)
