from pathlib import Path

from ray.util.actor_pool import ActorPool
from tqdm.auto import tqdm

from parse.ray_.stanza_ import StanzaActor
from parse.utils import get_parser

if __name__ == "__main__":
    parser = get_parser("**/*.conllu.gz")

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
    language = args.language

    in_paths: list[Path] = list(sorted(Path(in_root).glob(pattern)))
    out_paths: list[Path] = [
        out_root / p.relative_to(in_root).with_suffix("").with_suffix(".conllu.gz")
        for p in in_paths
    ]

    actors = []
    if args.num_gpu:
        actors.extend(
            StanzaActor.options(
                num_cpus=args.scale_cpu, num_gpus=args.scale_gpu
            ).remote(
                language,
                device="cuda",
                batch_size=args.batch_size,
                min_sentence_len=args.min_sentence_len,
                max_sentence_len=args.max_sentence_len,
            )
            for _ in range(args.num_gpu)
        )
    if args.num_cpu:
        actors.extend(
            StanzaActor.options(num_cpus=args.scale_cpu, num_gpus=0).remote(
                language,
                batch_size=args.batch_size,
                min_sentence_len=args.min_sentence_len,
                max_sentence_len=args.max_sentence_len,
            )
            for _ in range(args.num_cpu)
        )
    pool = ActorPool(actors)

    io_paths = [(in_path, out_path) for in_path, out_path in zip(in_paths, out_paths)]
    list(
        tqdm(
            pool.map_unordered(
                lambda a, io: a.parse.remote(*io),
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
