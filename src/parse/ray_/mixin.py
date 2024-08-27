import sys
import traceback

import ray
from ray.exceptions import RayActorError


def ray_remote_wrapper(
    cls: type | None = None,
    /,
    concurrency_groups: dict[str, int] | None = None,
    **kwargs,
):
    concurrency_groups = concurrency_groups or {
        "io": 2,
        "process": 1,
        "compute": 1,
    }

    def decorator(cls):
        cg_io = concurrency_groups.get("io", None)
        cg_process = concurrency_groups.get("process", None)
        cg_compute = concurrency_groups.get("compute", None)

        @ray.remote(concurrency_groups=concurrency_groups, **kwargs)
        class Decorated(cls):
            @ray.method(concurrency_group=cg_io)
            @staticmethod
            def read(*args, **kwargs):
                return cls.read(*args, **kwargs)

            @ray.method(concurrency_group=cg_io)
            @staticmethod
            def write(*args, **kwargs):
                return cls.write(*args, **kwargs)

            @ray.method(concurrency_group=cg_process)
            def process(self, *args, **kwargs):
                try:
                    return cls.process(self, *args, **kwargs)
                except Exception as e:
                    print(
                        (
                            f"{cls.__name__}.process failed with params:\n"
                            f"  args={args},\n"
                            f"  kwargs={kwargs}\n"
                        ),
                        file=sys.stderr,
                    )
                    traceback.print_exc()
                    raise RayActorError from e

            @ray.method(concurrency_group=cg_process)
            def pre_process(self, *args, **kwargs):
                return cls.pre_process(self, *args, **kwargs)

            @ray.method(concurrency_group=cg_compute)
            def parse(self, *args, **kwargs):
                return cls.parse(self, *args, **kwargs)

        return Decorated

    return decorator(cls) if cls is not None else decorator
