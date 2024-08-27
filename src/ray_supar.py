from pathlib import Path

from ray.util.actor_pool import ActorPool
from tqdm.auto import tqdm

from parse.ray_.supar_ import SuparActor
from parse.utils import get_parser

if __name__ == "__main__":
    parser = get_parser("**/*.conllu.gz")

    group_me_arch = parser.add_mutually_exclusive_group()
    group_me_arch.add_argument(
        "-a",
        "--arch",
        type=str,
        choices=["biaffine", "crf2o"],
        default="biaffine",
    )
    group_me_arch.add_argument(
        "--biaffine",
        action="store_const",
        const="biaffine",
        dest="arch",
    )
    group_me_arch.add_argument(
        "--crf2o",
        action="store_const",
        const="crf2o",
        dest="arch",
    )

    parser.add_argument(
        "--base_path",
        type=Path,
        default=None,
    )

    parser.add_argument(
        "-b",
        "--batch_size",
        type=int,
        default=5000,
    )

    args = parser.parse_args()

    in_root = args.in_root
    out_root = args.out_root
    pattern = args.pattern
    arch = args.arch
    language = args.language
    base_path = args.base_path

    in_paths: list[Path] = list(sorted(Path(in_root).glob(pattern)))
    out_paths: list[Path] = [
        out_root / p.relative_to(in_root).with_suffix("").with_suffix(".conllu.gz")
        for p in in_paths
    ]

    actors = []
    if args.num_gpu:
        actors.extend(
            SuparActor.options(num_cpus=args.scale_cpu, num_gpus=args.scale_gpu).remote(
                arch=arch,
                language=language,
                batch_size=args.batch_size,
                base_path=base_path,
                device="cuda:0",
            )
            for _ in range(args.num_gpu)
        )
    if args.num_cpu:
        actors.extend(
            SuparActor.options(num_cpus=args.scale_cpu, num_gpus=0).remote(
                arch=arch,
                language=language,
                batch_size=args.batch_size,
                base_path=base_path,
            )
            for _ in range(args.num_cpu)
        )
    pool = ActorPool(actors)

    io_paths = [(in_path, out_path) for in_path, out_path in zip(in_paths, out_paths)]
    list(
        tqdm(
            pool.map_unordered(
                lambda a, io: a.process.remote(*io),
                io_paths,
            ),
            total=len(in_paths),
            desc="Processing",
            mininterval=5.0,
            maxinterval=60.0,
            smoothing=0,
            ascii=True,
        )
    )
