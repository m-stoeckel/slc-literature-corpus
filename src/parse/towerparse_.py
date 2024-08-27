import sys
from pathlib import Path
from typing import Final, Literal

from stanza.models.common.doc import Word

from parse.stanza_ import ParserWithStanzaPreProcessor
from towerparse.tower import TowerParser as TowerParserModel

BASE_PATH: Final[Path] = Path.cwd().parent / "models/towerparse/"


class TowerParser(ParserWithStanzaPreProcessor):
    def __init__(
        self,
        language: Literal["en", "de"] = "en",
        batch_size: int = 128,
        base_path: Path = BASE_PATH,
        preprocess: bool = True,
        **kwargs,
    ):
        super().__init__(
            language=language,
            preprocess=preprocess,
            **kwargs,
        )
        self.batch_size = batch_size
        self._base_path = base_path
        self.nlp = TowerParserModel(self._get_model_path(), device=self.device)

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

    def process(self, path: Path, out: Path):
        document, filtered = super().pre_process(path)

        predictions = self.parse(
            [[word.text for word in sentence.words] for sentence in filtered.sentences]
        )

        for sentence, prediction in zip(filtered.sentences, predictions):
            for word, (_index, _token, governor, relation) in zip(
                sentence.words, prediction
            ):
                word: Word
                word.head = governor
                word.deprel = relation

        if self.drop_filtered:
            self.write(filtered, out)
        else:
            self.write(document, out)

        return path

    def parse(self, tokens: list[list[str]]) -> list[list[tuple[int, str, int, str]]]:
        return self.nlp.parse(
            self.language,
            tokens,
            batch_size=self.batch_size,
        )
