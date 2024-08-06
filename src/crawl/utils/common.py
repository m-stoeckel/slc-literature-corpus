import sys
from argparse import Namespace
from dataclasses import dataclass, field
from itertools import islice
from typing import Any, Final

from crawl.utils.gutenberg import obtain_urls


def _cpu_count():
    try:
        import multiprocessing

        count = multiprocessing.cpu_count()
        print(f"Number of CPUs available: {count}", file=sys.stderr)
        return count
    except Exception:
        print(
            "Could not load multiprocessing library, setting default CPUs to 4",
            file=sys.stderr,
        )
        return 4


CPU_COUNT: Final[int] = _cpu_count()


def batched(iterable, n):
    """
    Batch data into tuples of length n. The last batch may be shorter.

    >>> list[batched('ABCDEFG', 3)]
    ["ABC", "DEF", "G"]
    """
    if n < 1:
        raise ValueError("n must be at least one")
    it = iter(iterable)
    while batch := tuple(islice(it, n)):
        yield batch


@dataclass
class SubOutput:
    success: bool
    stdout: str = ""
    stderr: str = ""
    args: dict[str, Any] = field(default_factory=dict)


def parse_args_urls(args: Namespace) -> list[str]:
    if args.fetch or args.urls is None or not args.urls.exists():
        urls = obtain_urls(
            write=args.urls if args.fetch or args.urls is not None else None
        )
    else:
        urls = [line.strip() for line in args.urls.open("r").readlines()]
    return urls
