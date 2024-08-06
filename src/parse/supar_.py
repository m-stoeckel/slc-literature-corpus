from pathlib import Path
from typing import Final, Literal

import pgzip
import torch
from conllu import TokenList
from supar import Parser
from supar.utils.data import Dataset
from tqdm import tqdm

from parse.abc import ParserABC
from parse.utils import ConllFileHelper, batched, tokenlist_to_str

BASE_PATH: Final[Path] = Path.cwd().parent / "models/"


class SuparRunner(ParserABC):
    def __init__(
        self,
        arch: Literal["biaffine", "crf2o"] = "biaffine",
        language: Literal["en", "de"] = "en",
        batch_size: int = 128,
        base_path: Path = BASE_PATH,
        device: str | torch.device = "cpu",
        quiet: bool = False,
    ):
        device = torch.device(device)
        if device.type == "cuda":
            if not torch.cuda.is_available():
                raise ValueError("device.type == cuda, but no GPU is available")
            else:
                torch.cuda.set_device(device)

        self.arch = arch
        self.lang = language
        self._base_path = Path(base_path)

        self.nlp = Parser.load(self._get_model_path())
        self.batch_size = batch_size
        self.quiet = quiet

    def _get_model_path(self) -> Path:
        if self.arch not in {"biaffine", "crf2o"}:
            raise ValueError(f"Invalid architecture: {self.arch}")
        if self.lang not in {"en", "de"}:
            raise ValueError(f"Invalid language: {self.lang}")

        model_path = Path(self._base_path / f"{self.arch}/{self.lang}/model.pt")

        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")

        return model_path

    def parse(self, path: Path, out: Path):
        sentences = ConllFileHelper.read(path)
        if not sentences:
            return False

        annotations = []
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
                batch: list[TokenList]
                texts = [
                    tokenlist_to_str(
                        sentence,
                        tokenized=True,
                        expand_contractions=True,
                    )
                    for sentence in batch
                ]

                dataset: Dataset = self.nlp.predict(
                    texts,
                    lang=self.lang,
                    proj=False,
                    verbose=False,
                )

                for sentence, prediction in zip(batch, dataset):
                    sentence: TokenList

                    prediction_conllu = str(prediction).strip()

                    metadata_str = "\n".join(
                        f"# {key} = {value}" for key, value in sentence.metadata.items()
                    )
                    if metadata_str:
                        prediction_conllu = "\n".join((metadata_str, prediction_conllu))

                    annotations.append(prediction_conllu)

                tq.update(len(batch))

        out.parent.mkdir(parents=True, exist_ok=True)
        with pgzip.open(out, "wt", encoding="utf-8") as fp:
            fp.write("\n\n".join(annotations) + "\n")  # type: ignore

        return True
