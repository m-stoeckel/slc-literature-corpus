import sys
import traceback
from argparse import Namespace
from pathlib import Path
from typing import Any

import paramiko
from paramiko.ssh_exception import SSHException
from pqdm.threads import pqdm as pqdm_threads

from crawl.utils.common import CPU_COUNT, SubOutput, batched, parse_args_urls


class SshError(Exception):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args)
        self.kwargs = kwargs


def parse_args_hosts(args: Namespace):
    hosts = []
    for host_arg in args.hosts:
        if (p_host := Path(host_arg)).exists():
            hosts.extend([h.strip() for h in p_host.open("r").readlines()])
        else:
            hosts.extend(host_arg.split(","))
    return hosts


def check_hosts(
    hosts: list[str],
    user: str | None = None,
    port: int = 22,
    jobs: int = CPU_COUNT,
    private_key: paramiko.Ed25519Key | None = None,
):
    _l = len(hosts)
    hosts = list(
        filter(
            bool,
            pqdm_threads(
                [
                    {
                        "hostname": host,
                        "username": user,
                        "port": port,
                        "pkey": private_key,
                    }
                    for host in hosts
                ],
                return_online_host,
                n_jobs=jobs,
                argument_type="kwargs",
                leave=False,
            ),
        )
    )
    if len(hosts) == 0:
        raise RuntimeError("No hosts are online!")
    print(f"{len(hosts)}/{_l} hosts are online!", file=sys.stderr, flush=True)
    return hosts


def return_online_host(hostname: str, username=None, port=22, pkey=None):
    try:
        output = ssh_run_command(
            hostname,
            command=None,
            port=port,
            username=username,
            pkey=pkey,
        )
        if isinstance(output, SubOutput) and output.success:
            return hostname
        else:
            print(f"Host {hostname} is offline")
            return False
    except Exception as e:
        print(f"Error connecting to host {hostname}: {e}", file=sys.stderr, flush=True)
        return False


def ssh_run_command(
    hostname: str,
    command: str | None = None,
    username: str | None = None,
    port: int = 22,
    pkey: paramiko.PKey | None = None,
) -> SubOutput:
    hostname, username, ssh_client = ssh_connect(hostname, username)
    try:
        # trunk-ignore(bandit/B507)
        ssh_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())

        ssh_client.connect(
            hostname=hostname,
            port=port,
            username=username,
            pkey=pkey,
            timeout=5,
        )
        if command is not None:
            # trunk-ignore(bandit/B601)
            _stdin, stdout, stderr = ssh_client.exec_command(command)

            return SubOutput(
                True,
                stdout.read().decode().strip(),
                stderr.read().decode().strip(),
            )
        return SubOutput(True)
    except TimeoutError as e:
        if command is None:
            return SubOutput(False)
        else:
            return SubOutput(
                False,
                stderr=str(e),
                args={
                    "hostname": str(hostname),
                    "command": str(command),
                    "username": str(username),
                    "port": str(port),
                    "pkey": str(pkey),
                },
            )
    except SSHException:
        return SubOutput(
            False,
            stderr=traceback.format_exc(),
            args={
                "hostname": str(hostname),
                "command": str(command),
                "username": str(username),
                "port": str(port),
                "pkey": str(pkey),
            },
        )
    except Exception as e:
        raise SshError(
            f"Failed to connect to {hostname}:{port}",
            hostname=str(hostname),
            command=str(command),
            username=str(username),
            port=str(port),
            pkey=str(pkey),
        ) from e
    finally:
        ssh_client.close()


def ssh_connect(hostname, username):
    if username is None and "@" in hostname:
        username, hostname = hostname.split("@", 1)

    ssh_client = paramiko.SSHClient()
    return hostname, username, ssh_client


def crawl(args: Namespace):
    urls = parse_args_urls(args)

    url_batches = list(batched(urls, args.batch_size))

    hosts = parse_args_hosts(args)

    private_key = (
        paramiko.Ed25519Key.from_private_key_file(args.private_key)
        if args.private_key is not None
        else None
    )

    if args.check_hosts:
        hosts = check_hosts(
            hosts,
            args.user,
            args.port,
            args.jobs,
            private_key,
        )

    n_jobs = min(len(hosts), args.jobs) if args.limit else args.jobs

    def _loop_hosts_iter():
        while True:
            yield from hosts

    print(
        f"Running {len(url_batches)} tasks in {n_jobs} jobs on {len(hosts)} hosts",
        file=sys.stderr,
        flush=True,
    )

    kwargs: list[dict[str, Any]] = [
        {
            "hostname": host,
            "command": f'wget -P {args.output} -nc {" ".join(urls)}',
            "username": args.user,
            "port": args.port,
            "pkey": private_key,
        }
        for (urls, host) in zip(url_batches, _loop_hosts_iter())
    ]

    outputs: list[SubOutput] = pqdm_threads(
        kwargs,
        ssh_run_command,
        n_jobs=n_jobs,
        argument_type="kwargs",
        leave=False,
    )

    return outputs
