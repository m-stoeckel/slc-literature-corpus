import logging
import sys
from pathlib import Path
from typing import Final, Literal, Optional

import pgzip
import torch
from conllu import Token, TokenList
from stanza import Pipeline
from stanza.pipeline.core import DownloadMethod

from parse.abc import ParserABC
from parse.utils import (
    EMPTY_CONLLU_TOKEN,
    ConllFileHelper,
    handle_contractions,
    tokenlist_to_str,
)
from towerparse.tower import TowerParser

BASE_PATH: Final[Path] = Path.cwd().parent / "models/towerparse/"


class TowerParseRunner(ParserABC):
    tokenizer: Optional[Pipeline] = None

    def __init__(
        self,
        language: Literal["en", "de"] = "en",
        device: str | torch.device = "cpu",
        batch_size: int = 128,
        base_path: Path = BASE_PATH,
        tokenize: bool = True,
    ):
        device = torch.device(device)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise ValueError(f"device := {device}, but no GPU is available")

        self.batch_size = batch_size
        if tokenize:
            self.tokenizer = Pipeline(
                lang=language,
                processors="tokenize,mwt",
                tokenize_no_ssplit=True,
                download_method=DownloadMethod.REUSE_RESOURCES,
                device="cpu",
                depparse_batch_size=batch_size,
            )

        self.lang = language
        self._base_path = base_path
        self.nlp = TowerParser(self._get_model_path(), device=device)

    def _get_model_path(self) -> Path:
        match self.lang:
            case "en":
                model_path = Path(self._base_path / "UD_English-EWT")
            case "de":
                model_path = Path(self._base_path / "UD_German-HDT")
            case invalid:
                raise ValueError(f"Invalid language: {invalid}")

        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")

        print(f"Loding model from {model_path}")
        return model_path

    def parse(self, path: Path, out: Path):
        sentences = ConllFileHelper.read(path)
        Path(out).parent.mkdir(parents=True, exist_ok=True)

        if self.tokenizer is not None:
            tokens: list[list[str]] = [
                [
                    token.text
                    for token in self.tokenizer(
                        tokenlist_to_str(sentence).replace("ſ", "s")
                    )
                    .sentences[0]
                    .tokens
                ]
                for sentence in sentences
            ]
        else:
            tokens = [
                [
                    token["form"]
                    for token in handle_contractions(sentence, expand=True)
                ]
                for sentence in sentences
            ]
        metadata = [sentence.metadata for sentence in sentences]

        try:
            predictions = self.nlp.parse(
                self.lang,
                tokens,
                batch_size=self.batch_size,
            )

            sentences = [
                TokenList(
                    [
                        Token(
                            EMPTY_CONLLU_TOKEN
                            | {
                                "id": index,
                                "form": token,
                                "head": governor,
                                "deprel": relation,
                            }
                        )
                        for (index, token, governor, relation) in prediction
                    ],
                    metadata=meta,
                )
                for prediction, meta in zip(predictions, metadata)
            ]

            with pgzip.open(out, "wt", encoding="utf-8") as fp:
                fp.writelines(sentence.serialize() for sentence in sentences)  # type: ignore
        except IndexError:
            print(tokens, file=sys.stderr)
            raise

        return True
