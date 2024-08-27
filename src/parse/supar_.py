from pathlib import Path
from typing import Final, Literal

import torch
from stanza.models.common.doc import Sentence
from supar import Parser
from supar.models.dep.biaffine.transform import CoNLLSentence
from supar.utils.data import Dataset
from tqdm import tqdm

from parse.stanza_ import ParserWithStanzaPreProcessor
from parse.utils import batched

BASE_PATH: Final[Path] = Path.cwd().parent / "models/"


class SuparParser(ParserWithStanzaPreProcessor):
    def __init__(
        self,
        arch: Literal["biaffine", "crf2o"] = "biaffine",
        language: Literal["en", "de"] = "en",
        batch_size: int = 128,
        base_path: Path = BASE_PATH,
        device: str | torch.device = "cpu",
        quiet: bool = False,
    ):
        super().__init__(language=language, device=device)

        if self.device.type == "cuda":
            torch.cuda.set_device(self.device)

        self.arch = arch
        self._base_path = Path(base_path)

        self.nlp = Parser.load(self._get_model_path())
        self.batch_size = batch_size
        self.quiet = quiet

    def _get_model_path(self) -> Path:
        if self.arch not in {"biaffine", "crf2o"}:
            raise ValueError(f"Invalid architecture: {self.arch}")
        if self.language not in {"en", "de"}:
            raise ValueError(f"Invalid language: {self.language}")

        model_path = Path(self._base_path / f"{self.arch}/{self.language}/model.pt")

        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")

        return model_path

    def process(self, path: Path, out: Path):
        path, out = Path(path), Path(out)

        document, filtered = super().pre_process(path)
        sentences: list[Sentence] = filtered.sentences

        if not sentences:
            raise ValueError(f"No sentences found in {path.name}")

        with tqdm(
            total=len(sentences),
            desc=f"Parsing: {path.name}",
            position=1,
            leave=False,
            ascii=True,
            smoothing=0,
            disable=self.quiet,
        ) as tq:
            for batch in batched(sentences, self.batch_size):
                dataset: Dataset = self.parse(
                    [[word.text for word in sentence.words] for sentence in batch]
                )

                for sentence, prediction in zip(batch, dataset):
                    sentence: Sentence
                    prediction: CoNLLSentence

                    arcs = prediction.values[prediction.maps.get("arcs", 6)]
                    rels = prediction.values[prediction.maps.get("rels", 7)]

                    if prediction.maps.get("tags") is None:
                        for word, head, deprel in zip(sentence.words, arcs, rels):
                            word.head = head
                            word.deprel = deprel
                    else:
                        tags = prediction.values[prediction.maps["tags"]]
                        for word, head, deprel, tag in zip(
                            sentence.words, arcs, rels, tags
                        ):
                            word.head = head
                            word.deprel = deprel
                            word.upos = tag

                tq.update(len(batch))

        if self.drop_filtered:
            self.write(filtered, out)
        else:
            self.write(document, out)

        return path

    def parse(self, texts: list[list[str]]) -> Dataset:
        # NOTE: we do not set `lang` parameter here, as we are already passing words
        return self.nlp.predict(
            texts,
            lang=None,
            proj=False,
            verbose=False,
        )
