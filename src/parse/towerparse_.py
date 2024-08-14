import sys
from pathlib import Path
from typing import Final, Literal

import torch
from stanza.models.common.doc import Document, Word

from parse.stanza_ import ParserWithStanzaPreProcessor
from parse.utils import ConllFileHelper
from towerparse.tower import TowerParser

BASE_PATH: Final[Path] = Path.cwd().parent / "models/towerparse/"


class TowerParseRunner(ParserWithStanzaPreProcessor):
    def __init__(
        self,
        language: Literal["en", "de"] = "en",
        batch_size: int = 128,
        base_path: Path = BASE_PATH,
        preprocess: bool = True,
        device: str | torch.device = "cpu",
    ):
        super().__init__(
            language=language,
            preprocess=preprocess,
            device=device,
        )
        self.batch_size = batch_size
        self._base_path = base_path
        self.nlp = TowerParser(self._get_model_path(), device=self.device)

    def _get_model_path(self) -> Path:
        match self.language:
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
        document: Document = self.read(path)
        tokens = [
            [word.text for word in sentence.words] for sentence in document.sentences
        ]

        Path(out).parent.mkdir(parents=True, exist_ok=True)

        try:
            predictions = self.nlp.parse(
                self.language,
                tokens,
                batch_size=self.batch_size,
            )

            for sentence, prediction in zip(document.sentences, predictions):
                for word, (_index, _token, governor, relation) in zip(
                    sentence.words, prediction
                ):
                    word: Word
                    word.head = governor
                    word.deprel = relation

            ConllFileHelper.write_stanza(document, out)
        except IndexError:
            print(tokens, file=sys.stderr)
            raise

        return True
