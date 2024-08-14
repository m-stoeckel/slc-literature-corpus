from abc import ABC, abstractmethod
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Literal

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


class ParserABC(ABC):
    def __init__(
        self,
        language: Literal["en", "de"] = "en",
        device: str | torch.device = "cpu",
    ) -> None:
        super().__init__()
        self.language = language

        self.device = torch.device(device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise ValueError(f"device := {self.device}, but no GPU is available")

    @abstractmethod
    def parse(self, path: Path, out: Path):
        raise NotImplementedError
