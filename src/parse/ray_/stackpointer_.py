import ray

from parse.stackpointer_ import StackPointerRunner


@ray.remote(num_cpus=1, num_gpus=1, max_restarts=3, max_task_retries=-1)
class StackPointerActor(StackPointerRunner):
    def __init__(
        self,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs.setdefault("quiet", True))
