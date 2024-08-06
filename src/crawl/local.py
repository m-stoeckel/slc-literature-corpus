import traceback
from argparse import Namespace
from pathlib import Path
from urllib.parse import unquote, urlparse

import requests
from pqdm.threads import pqdm as pqdm_threads

from crawl.utils.common import SubOutput, parse_args_urls


def run_local(url: str, path: Path) -> SubOutput:
    try:
        response = requests.get(url, timeout=30)

        if response.status_code != 200:
            return SubOutput(
                False,
                args={
                    "url": url,
                    "status": response.status_code,
                    "message": response.text,
                },
            )

        with open(path, "wb") as f:
            f.write(response.content)

        return SubOutput(True)
    except requests.RequestException as e:
        return SubOutput(
            False,
            args={
                "url": url,
                "exception": str(e),
                "traceback": traceback.format_exc(),
            },
        )
    except Exception as e:
        return SubOutput(
            False,
            args={
                "url": url,
                "exception": str(e),
                "traceback": traceback.format_exc(),
            },
        )


def crawl(args: Namespace) -> list[SubOutput]:
    urls: list[str] = parse_args_urls(args)
    output: Path = args.output
    n_jobs: int = args.n_jobs

    kwargs: list[dict[str, str | Path]] = [
        {
            "url": url,
            "path": output / unquote(urlparse(url).path).split("/")[-1],
        }
        for url in urls
    ]
    outputs = pqdm_threads(kwargs, run_local, n_jobs=n_jobs, argument_type="kwargs")

    return outputs
