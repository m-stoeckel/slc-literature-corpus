from argparse import ArgumentParser, Namespace
from pathlib import Path

from tqdm import tqdm

from parse.abc import TaskABC
from parse.spacy_ import SpacyRunner


class SpacyTask(TaskABC):
    @classmethod
    def get_argparser(cls) -> ArgumentParser:
        parser = ArgumentParser()
        parser.add_argument(
            "paths",
            type=Path,
            nargs="+",
            metavar="[INPUT_FILE ...] OUTPUT_DIR",
        )
        parser.add_argument(
            "--pattern",
            type=str,
            default=None,
            # default="*.conllu.gz",
        )

        group_me_language = parser.add_mutually_exclusive_group()
        group_me_language.add_argument(
            "-l",
            "--language",
            type=str,
            choices=["en_core_web_sm", "de_core_news_sm"],
            default="en_core_web_sm",
        )
        group_me_language.add_argument(
            "--en",
            action="store_const",
            const="en_core_web_sm",
            dest="language",
        )
        group_me_language.add_argument(
            "--de",
            action="store_const",
            const="de_core_news_sm",
            dest="language",
        )

        parser.add_argument(
            "--parser",
            action="store_true",
            help="Enable dependency parser",
            default=False,
        )

        return parser

    @classmethod
    def get_paths(cls, args: Namespace) -> tuple[list[Path], list[Path]]:
        paths: list[Path] = args.paths
        in_paths, out_path = paths[:-1], paths[-1]

        if args.pattern is not None and args.pattern:
            if len(in_paths) > 1:
                raise ValueError(
                    "Invalid input paths: --pattern can only be set with a single input path."
                )
            in_root = in_paths[0]
            in_paths: list[Path] = list(sorted(Path(in_root).glob(args.pattern)))

        for in_path in in_paths:
            if not in_path.exists():
                raise FileNotFoundError(f"{in_path} does not exist.")
            elif in_path.is_dir():
                raise IsADirectoryError(
                    f"Input path is a directory, not a file: {in_path}"
                )

        # Prepare output paths.
        # Path.relative_to() will raise an error if the paths are not in the same tree.
        out_paths: list[Path] = [
            out_path / (in_path.with_suffix("").with_suffix(".conllu.gz").name)
            for in_path in in_paths
        ]

        if not out_path.exists():
            out_path.mkdir(parents=True, exist_ok=True)
        elif not out_path.is_dir():
            raise NotADirectoryError(f"{out_path} exists, but is not a directory")
        return in_paths, out_paths

    def process(
        self,
        in_paths: list[Path],
        out_paths: list[Path],
        language: str = "en",
        parser: bool = False,
        **kwargs,
    ):
        runner = SpacyRunner(language, parser=parser, validate=True)
        for in_path, out_path in zip(
            tqdm(in_paths, desc="Spacy", smoothing=0), out_paths
        ):
            runner.parse(in_path, out_path)


if __name__ == "__main__":
    SpacyTask().run()
