from pathlib import Path

from ray.util.actor_pool import ActorPool
from tqdm.auto import tqdm

from parse.ray_.spacy_ import SpacyActor
from parse.utils import get_parser

if __name__ == "__main__":
    parser = get_parser("**/*.xmi.bz2")

    lang_map = {
        "en": "en_core_web_sm",
        "de": "de_core_news_sm",
    }
    args = parser.parse_args()

    in_root = args.in_root
    out_root = args.out_root
    pattern = args.pattern
    language = lang_map[args.language]

    in_paths: list[Path] = list(sorted(Path(in_root).glob(pattern)))
    out_paths: list[Path] = [
        out_root / p.relative_to(in_root).with_suffix("").with_suffix(".conllu.gz")
        for p in in_paths
    ]

    actors = []
    if args.num_gpu:
        actors.extend(
            SpacyActor.options(num_cpus=args.scale_cpu, num_gpus=args.scale_gpu).remote(
                language
            )
            for _ in range(args.num_gpu)
        )
    if args.num_cpu:
        actors.extend(
            SpacyActor.options(num_cpus=args.scale_cpu, num_gpus=0).remote(language)
            for _ in range(args.num_cpu)
        )
    pool = ActorPool(actors)

    list(
        tqdm(
            pool.map_unordered(
                lambda a, io: a.parse.remote(*io),
                zip(in_paths, out_paths),
            ),
            total=len(in_paths),
            desc="Processing",
            mininterval=5.0,
            smoothing=0,
            ascii=True,
            maxinterval=60.0,
        )
    )
