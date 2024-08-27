from abc import ABC, abstractmethod
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Any, Literal, TypeVar

import torch


class TaskABC(ABC):
    def run(self):
        args = self.get_argparser().parse_args()
        in_paths, out_paths = self.get_paths(args)
        self.process(in_paths, out_paths, **vars(args))

    @classmethod
    @abstractmethod
    def get_argparser(cls) -> ArgumentParser:
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def get_paths(cls, args: Namespace) -> tuple[list[Path], list[Path]]:
        raise NotImplementedError

    @abstractmethod
    def process(self, in_paths: list[Path], out_paths: list[Path], **kwargs):
        raise NotImplementedError


T_doc = TypeVar("T_doc")


class ParserABC(ABC):
    def __init__(
        self,
        language: Literal["en", "de"] = "en",
        device: str | torch.device = "cpu",
        min_sentence_len: int = -1,
        max_sentence_len: int = -1,
        drop_filtered: bool = False,
        **kwargs,
    ) -> None:
        super().__init__()
        self.language = language

        self.device = torch.device(device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise ValueError(f"device := {self.device}, but no GPU is available")

        self.min_sentence_len = min_sentence_len
        self.max_sentence_len = max_sentence_len
        self.drop_filtered = drop_filtered

    @staticmethod
    @abstractmethod
    def read(path: Path) -> Any:
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def write(document: Any, path: Path) -> None:
        raise NotImplementedError

    @abstractmethod
    def process(self, path: Path, out: Path) -> Path:
        raise NotImplementedError

    @abstractmethod
    def filter(
        self,
        document: T_doc,
    ) -> T_doc:
        raise NotImplementedError
