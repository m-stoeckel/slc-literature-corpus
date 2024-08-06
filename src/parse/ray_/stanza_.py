import ray

from parse.stanza_ import StanzaRunner


@ray.remote(num_cpus=1, num_gpus=1, max_restarts=3, max_task_retries=-1)
class StanzaActor(StanzaRunner):
    def __init__(
        self,
        *args,
        quiet: bool = True,
        **kwargs,
    ):
        super().__init__(*args, **kwargs, quiet=quiet)
