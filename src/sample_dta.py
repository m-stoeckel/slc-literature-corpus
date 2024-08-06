from argparse import ArgumentParser
from pathlib import Path

from tqdm import tqdm

from parse.utils import sample_from_conllu

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "paths",
        type=Path,
        nargs="+",
        metavar="[INPUT_DIR ...] OUTPUT_DIR",
        help="Input directories containing CoNLL-U files and a single output directory for sampled CoNLL-U files. Input paths must be directories. Output directory will be create if it does not exist.",
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default="*.conllu.gz",
        help=("glob pattern to match CoNLL-U files. " "Default: '*.conllu.gz'."),
    )

    parser.add_argument(
        "-k",
        "--num_sentences",
        type=int,
        default=450,
        help=(
            "Maximum number of sentences to sample from each input directory. "
            "Default: 450."
        ),
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help=(
            "Seed for random sampling. "
            "All input files will be shuffled with this seed. "
            "Default: 42."
        ),
    )
    parser.add_argument(
        "--lengths",
        type=int,
        nargs="+",
        default=[5, 10, 15, 20, 30, 40, 50, 60, 70],
    )
    parser.add_argument(
        "--period",
        type=int,
        default=3,
    )

    group_by_decades = parser.add_argument(
        "--group-by-decades",
        dest="group_by_decades",
        default="89",
        help=(
            "Which decades to include, per --group-by-regex. "
            "This value will be formatted into the regex pattern, if a '%s' is present. "
            "Default: 89."
        ),
    )
    group_regex = parser.add_mutually_exclusive_group(required=True)
    group_regex.add_argument(
        "-gbr",
        "--group-by-regex",
        dest="group_by_regex",
        type=str,
    )
    group_regex.add_argument(
        "-gby",
        "--group-by-year",
        dest="group_by_regex",
        action="store_const",
        const=r".*\/(?P<a_prefix>[^/]+\/[^/]+\/)[^/]+_(?P<b_date>1[%s]\d\d)[^/]+",
    )
    group_regex.add_argument(
        "-gbyF",
        "--group-by-year-flatten",
        dest="group_by_regex",
        action="store_const",
        const=r".*\/[^/]+\/(?P<a_prefix>[^/]+\/)[^/]+_(?P<b_date>1[%s]\d\d)[^/]+",
        help="Group by year, flatten the first subdirectory (Kernkorpus/Erweiterungstexte).",
    )
    group_regex.add_argument(
        "-gbyf",
        "--group-by-year-flattend",
        dest="group_by_regex",
        action="store_const",
        const=r".*\/(?P<a_prefix>[^/]+\/)[^/]+_(?P<b_date>1[%s]\d\d)[^/]+",
        help="Group by year, requires flattend input.",
    )
    group_regex.add_argument(
        "-gbd",
        "--group-by-decade",
        dest="group_by_regex",
        action="store_const",
        const=r".*\/(?P<a_prefix>[^/]+\/[^/]+\/)[^/]+_(?P<b_date>1[%s]\d)\d[^/]+",
    )
    group_regex.add_argument(
        "-gbdF",
        "--group-by-decade-flatten",
        dest="group_by_regex",
        action="store_const",
        const=r".*\/[^/]+\/(?P<a_prefix>[^/]+\/)[^/]+_(?P<b_date>1[%s]\d)\d[^/]+",
    )
    group_regex.add_argument(
        "-gbdf",
        "--group-by-decade-flattend",
        dest="group_by_regex",
        action="store_const",
        const=r".*\/(?P<a_prefix>[^/]+\/)[^/]+_(?P<b_date>1[%s]\d)\d[^/]+",
    )

    parser.add_argument(
        "--group-exhaustive",
        action="store_true",
        help="Require all files to match the group pattern. Default: False.",
    )

    args = parser.parse_args()

    in_paths, out_path = args.paths[:-1], args.paths[-1]

    for in_path in in_paths:
        if not in_path.is_dir():
            raise NotADirectoryError(f"Input path is not a directory: {in_path}")

    if not out_path.exists():
        out_path.mkdir(parents=True, exist_ok=True)
    elif not out_path.is_dir():
        raise NotADirectoryError(
            f"Output path exists, but is not a directory: {out_path}"
        )

    group_by_regex = args.group_by_regex
    if "%s" in group_by_regex:
        group_by_regex = group_by_regex % args.group_by_decades

    for folder in tqdm(in_paths, "Sampling", smoothing=0):
        sample_from_conllu(
            folder,
            out_path,
            k=args.num_sentences,
            pattern=args.pattern,
            seed=args.seed,
            lengths=args.lengths,
            period=args.period,
            group_by_regex=group_by_regex,
            group_all_match=args.group_exhaustive,
        )
