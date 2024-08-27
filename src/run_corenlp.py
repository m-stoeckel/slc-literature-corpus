import threading
from argparse import ArgumentParser, Namespace
from pathlib import Path
from queue import PriorityQueue, Queue

import pgzip
from tqdm import tqdm

from parse.abc import TaskABC
from parse.corenlp_ import CoreNlpServer, CoreNlpWorker
from parse.utils import ConllFileHelper, conllu_tokenlist_to_str


class TqdmQueue(Queue):
    def __init__(self, *args, maxsize: int = 0, **kwargs) -> None:
        super().__init__(maxsize=maxsize)
        self.tq = tqdm(*args, **kwargs)

    def task_done(self) -> None:
        self.tq.update(1)
        return super().task_done()

    def reset(self, total: float | None = None) -> None:
        self.tq.reset(total=total)


class CoreNlpTask(TaskABC):
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
        )

        group_me_language = parser.add_mutually_exclusive_group()
        group_me_language.add_argument(
            "-l",
            "--language",
            type=str,
            choices=["en", "de"],
            default="en",
        )
        group_me_language.add_argument(
            "--en",
            action="store_const",
            const="en",
            dest="language",
        )
        group_me_language.add_argument(
            "--de",
            action="store_const",
            const="de",
            dest="language",
        )

        parser.add_argument(
            "--no-start",
            action="store_false",
            dest="start",
            help="Do not start the server",
        )

        parser.add_argument(
            "-j",
            "--n_jobs",
            type=int,
            default=4,
            help="number of worker threads",
        )
        parser.add_argument(
            "--memory",
            type=int,
            default=16,
            help="memory per server in GB",
        )
        parser.add_argument("--port", type=int, default=23023)

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

            out_paths: list[Path] = [
                out_path
                / (
                    in_path.with_suffix("")
                    .with_suffix(".conllu.gz")
                    .relative_to(in_root)
                )
                for in_path in in_paths
            ]
        else:
            out_paths: list[Path] = [
                out_path / (in_path.with_suffix("").with_suffix(".conllu.gz").name)
                for in_path in in_paths
            ]

        for in_path in in_paths:
            if not in_path.exists():
                raise FileNotFoundError(f"{in_path} does not exist.")
            elif in_path.is_dir():
                raise IsADirectoryError(
                    f"Input path is a directory, not a file: {in_path}"
                )

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
        port: int = 23023,
        memory: int = 128,
        n_jobs: int = 16,
        start: bool = True,
        **kwargs,
    ):
        n_jobs = n_jobs if n_jobs and n_jobs > 0 else 1

        if start:
            self.__run_force_start(in_paths, out_paths, language, port, memory, n_jobs)
        else:
            self.__run(in_paths, out_paths, n_jobs, port)

    def __run_force_start(
        self,
        in_paths: list[Path],
        out_paths: list[Path],
        language: str = "en",
        port: int = 23023,
        memory: int = 128,
        n_jobs: int = 16,
    ):
        language = {"en": "english", "de": "german"}[language]

        with CoreNlpServer(
            language=language,
            threads=n_jobs,
            port=port,
            memory=memory,
        ) as client:
            client.ensure_alive()
            self.__run(in_paths, out_paths, n_jobs, port)

    def __run(
        self,
        in_paths: list[Path],
        out_paths: list[Path],
        n_jobs: int,
        port: int,
    ):
        self.task_queue = TqdmQueue(
            total=0,
            desc="Processing",
            position=1,
            leave=False,
            smoothing=0,
        )
        self.result_queue = PriorityQueue()

        threads: list[threading.Thread] = []
        for _ in range(n_jobs):
            thread = threading.Thread(
                target=CoreNlpWorker.process,
                args=(self.task_queue, self.result_queue),
                kwargs={"port": port},
                daemon=True,
            )
            thread.start()
            threads.append(thread)

        with tqdm(total=len(in_paths), desc="Files", position=0) as tq:
            for in_path, out_path in zip(in_paths, out_paths):
                tq.set_postfix(file=in_path.name)
                self.__process(in_path, out_path)
                tq.update(1)

    def __process(self, in_path: Path, out_path: Path):
        out_path.parent.mkdir(parents=True, exist_ok=True)
        sentences = ConllFileHelper.read_conllu(in_path)

        n_added_sentences = 0
        self.task_queue.reset(total=len(sentences))
        try:
            for idx, sentence in enumerate(sentences, start=1):
                text: str = conllu_tokenlist_to_str(
                    sentence,
                    tokenized=True,
                    expand_contractions=True,
                )
                metadata: str = "\n".join(
                    f"# {key} = {value}" for key, value in sentence.metadata.items()
                )
                if not text.strip():
                    continue
                self.task_queue.put_nowait((idx, text, metadata))
                n_added_sentences += 1

            self.task_queue.join()
        except Exception as e:
            raise threading.ThreadError from e

        with pgzip.open(out_path, "wt", encoding="utf-8") as fp:
            while not self.result_queue.empty():
                _, result = self.result_queue.get_nowait()
                fp.write(result.strip() + "\n\n")  # type: ignore


if __name__ == "__main__":
    CoreNlpTask().run()
