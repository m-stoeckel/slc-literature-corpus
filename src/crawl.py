import sys
from argparse import Namespace
from pathlib import Path

from crawl.ssh import SshError
from crawl.utils.common import CPU_COUNT, SubOutput

if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()
    subparsers = parser.add_subparsers(dest="mode")

    common_parser = ArgumentParser(add_help=False)
    common_parser.add_argument("output", type=Path)
    common_parser.add_argument(
        "--urls",
        type=Path,
        default=None,
        metavar="URLS_FILE",
        help="File containing URLs to fetch. If not available, set --fetch to fetch URLs from the Gutenberg website. If --fetch was given, will write processed URLs to this file and overwrite it if it already exists. Writing the URLs can be disabled with --no-write. Default: None.",
    )
    common_parser.add_argument(
        "--type",
        type=str,
        default="txt",
        choices=["txt", "htm"],
        help="File type to fetch. Default: txt",
    )
    fetch_group = common_parser.add_mutually_exclusive_group()
    fetch_group.add_argument(
        "--fetch",
        action="store_true",
        default=False,
        help="Fetch the Gutenberg URLs via the ls-R file, process them, and write the output to the --urls file (if given). Note: Writing the URLs be disabled with --no-write.",
    )
    fetch_group.add_argument(
        "--no-fetch",
        dest="fetch",
        action="store_false",
        help="Disable fetching the Gutenberg URLs via the ls-R file.",
    )
    write_group = common_parser.add_mutually_exclusive_group()
    write_group.add_argument(
        "--write",
        dest="write",
        action="store_true",
        default=True,
        help="Enable writing fetched URLs to the --urls file.",
    )
    write_group.add_argument(
        "--no-write",
        dest="write",
        action="store_false",
        help="Disable writing fetched URLs to the --urls file.",
    )
    common_parser.add_argument(
        "--update",
        action="store_true",
        default=False,
        help="Update the Gutenberg URLs file with the latest URLs. Combines --fetch and --write.",
    )

    local_parser = subparsers.add_parser(
        "local", allow_abbrev=True, parents=[common_parser]
    )
    local_parser.add_argument("-j", "--jobs", type=int, default=4)

    ssh_parser = subparsers.add_parser("ssh")
    ssh_parser.add_argument("-j", "--jobs", type=int, default=CPU_COUNT)
    ssh_parser.add_argument("--hosts", type=str, action="append")
    ssh_parser.add_argument(
        "--check-hosts", dest="check_hosts", action="store_true", default=True
    )
    ssh_parser.add_argument(
        "--no-check-hosts", dest="check_hosts", action="store_true", default=False
    )
    ssh_parser.add_argument(
        "--private_key",
        type=Path,
        default=(
            p_key_file
            if (p_key_file := Path.home() / ".ssh/id_ed25519").exists()
            else None
        ),
    )
    ssh_parser.add_argument("--user", type=str, default=None)
    ssh_parser.add_argument("--port", type=int, default=22)
    ssh_parser.add_argument(
        "--batch",
        "--batch-size",
        dest="batch_size",
        type=int,
        default=500,
    )
    ssh_parser.add_argument(
        "--no-limit",
        dest="limit",
        action="store_false",
        default=True,
        help="Don't limit the number of jobs to the number of hosts",
    )

    args: Namespace = parser.parse_args()
    if args.update:
        args.fetch = True
        args.write = True
    print(args)

    if not args.fetch and (args.urls is None or not args.urls.exists()):
        raise ValueError(
            f"URLs file {args.urls} does not exist and --fetch is not set\n"
            f"Set --fetch to fetch the URLs or provide a valid file path"
        )

    match args.mode:
        case "local":
            from crawl.local import crawl as crawl_local

            outputs = crawl_local(args)
        case "ssh":
            from crawl.ssh import crawl as crawl_remote

            outputs = crawl_remote(args)
        case _:
            raise ValueError(f"Invalid mode: {args.mode}")

    any_error = False
    for o in outputs:
        if isinstance(o, SubOutput):
            if not o.success:
                stdout = "  \n".join(o.stdout.splitlines())
                stderr = "  \n".join(o.stderr.splitlines())
                print(
                    f"Error running command '{o.args['command']}'\n"
                    f"on host '{o.args['user']}@{o.args['host']}:{o.args['port']}'\n"
                    f"stdout:\n  {stdout}\n\n"
                    f"stderr:\n  {stderr}\n",
                    file=sys.stderr,
                )
                any_error = True
        else:
            if isinstance(o, SshError):
                raise RuntimeError(f"Some commands failed!\nargs:\n{o.kwargs}") from o  # type: ignore
            any_error = True

    if any_error:
        print(file=sys.stderr, flush=True)
        raise RuntimeError("Some commands failed!")
