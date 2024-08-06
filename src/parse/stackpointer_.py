import json
from pathlib import Path
from typing import Final, Generator, Literal, NamedTuple

import numpy as np
import pgzip
import torch
from attr import dataclass
from conllu import Metadata, Token, TokenList
from tqdm import tqdm

from neuronlp2.io import conllx_data
from neuronlp2.io.common import (
    DIGIT_RE,
    MAX_CHAR_LENGTH,
    PAD_ID_CHAR,
    PAD_ID_TAG,
    PAD_ID_WORD,
    ROOT,
    ROOT_CHAR,
    ROOT_POS,
)
from neuronlp2.models import StackPtrNet
from parse.abc import ParserABC
from parse.utils import EMPTY_CONLLU_TOKEN, ConllFileHelper, handle_contractions

BASE_PATH: Final[Path] = Path.cwd().parent / "models/stackpointer/"


class MinMax:
    def __init__(self, value: int = 0):
        self.min: int = value
        self.max: int = value

    def update(self, value):
        self.max = max(self.max, value)
        self.min = min(self.min, value)


class DataPoint(NamedTuple):
    word_ids: list[int]
    char_seq_ids: list[list[int]]
    pos_ids: list[int]
    forms: list[str]
    lemmata: list[str]
    metadata: dict | Metadata

    @classmethod
    def new(cls):
        return cls([0], [[1]], [1], [], [], {})


@dataclass
class StackedData:
    WORD: np.ndarray
    CHAR: np.ndarray
    POS: np.ndarray
    MASK_ENC: np.ndarray
    LENGTH: np.ndarray
    FORM: list[list[str]]
    LEMMA: list[list[str]]
    META: list[Metadata]

    def __getitem__(self, key) -> np.ndarray | list[Metadata]:
        if not hasattr(self, key):
            raise KeyError(key)
        return getattr(self, key)


@dataclass
class TensorData:
    WORD: torch.Tensor
    CHAR: torch.Tensor
    POS: torch.Tensor
    MASK_ENC: torch.Tensor
    LENGTH: torch.Tensor
    FORM: list[list[str]]
    LEMMA: list[list[str]]
    META: list[Metadata]

    def __getitem__(self, key) -> torch.Tensor | list[Metadata]:
        if not hasattr(self, key):
            raise KeyError(key)
        return getattr(self, key)

    def to(self, device: str | torch.device) -> "TensorData":
        self.WORD = self.WORD.to(device)
        self.CHAR = self.CHAR.to(device)
        self.POS = self.POS.to(device)
        self.MASK_ENC = self.MASK_ENC.to(device)
        self.LENGTH = self.LENGTH.to(device)
        return self


class StackPointerRunner(ParserABC):
    def __init__(
        self,
        language: Literal["en", "de"] = "en",
        device: str | torch.device = "cpu",
        batch_size: int = 128,
        beam: int = 1,
        base_path: Path = BASE_PATH,
        quiet: bool = False,
    ):
        self.device = torch.device(device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise ValueError("device.type == 'cuda', but no GPU is available")

        self.batch_size = batch_size
        self.beam = beam
        self.quiet = quiet

        model_dir = Path(base_path) / language
        for sub in ("alphabets", "config.json", "model.pt"):
            if not (model_dir / sub).exists():
                raise FileNotFoundError(
                    f"StackPointer file not found: {model_dir / sub}"
                )

        (
            self.word_alphabet,
            self.char_alphabet,
            self.pos_alphabet,
            self.type_alphabet,
        ) = conllx_data.create_alphabets(model_dir / "alphabets", None)

        config = json.load((model_dir / "config.json").open("r"))
        self.prior_order = config["prior_order"]
        self.nlp = StackPtrNet(
            word_dim=config["word_dim"],
            num_words=self.word_alphabet.size(),
            char_dim=config["char_dim"],
            num_chars=self.char_alphabet.size(),
            pos_dim=config["pos_dim"],
            num_pos=self.pos_alphabet.size(),
            rnn_mode=config["rnn_mode"],
            hidden_size=config["hidden_size"],
            encoder_layers=config["encoder_layers"],
            decoder_layers=config["decoder_layers"],
            num_labels=self.type_alphabet.size(),
            arc_space=config["arc_space"],
            type_space=config["type_space"],
            prior_order=self.prior_order,
            activation=config["activation"],
            p_in=config["p_in"],
            p_out=config["p_out"],
            p_rnn=config["p_rnn"],
            pos=config["pos"],
            grandPar=config["grandPar"],
            sibling=config["sibling"],
        )
        self.nlp = self.nlp.to(self.device)
        self.nlp.load_state_dict(
            torch.load(model_dir / "model.pt", map_location=self.device)
        )
        self.nlp.eval()

    def parse(self, path: Path, out: Path):
        sentences = self.read(path)

        out.parent.mkdir(parents=True, exist_ok=True)
        annotated = []
        with tqdm(
            total=len(sentences),
            desc=f"Parsing: {path.name}",
            position=1,
            leave=False,
            ascii=True,
            smoothing=0,
            disable=self.quiet,
        ) as tq:
            for batch in iterate_batch(
                self.stack(sentences),
                len(sentences),
                self.batch_size,
            ):
                with torch.no_grad():
                    batch_d = batch.to(self.device)
                    heads, types = self.nlp.decode(
                        batch_d.WORD,
                        batch_d.CHAR,
                        batch_d.POS,
                        mask=batch_d.MASK_ENC,
                        beam=self.beam,
                    )
                batch = batch_d.to("cpu")

                for i, length in enumerate(batch.LENGTH.numpy()):
                    sample_heads = heads[i][1 : length + 1]
                    sample_types = types[i][1 : length + 1]
                    sample_meta = batch.META[i]

                    annotated.append(
                        TokenList(
                            [
                                Token(
                                    EMPTY_CONLLU_TOKEN
                                    | {
                                        "id": index,
                                        "form": form,
                                        "lemma": lemma,
                                        "head": governor,
                                        "deprel": self.type_alphabet.get_instance(
                                            relation
                                        ),
                                    }
                                )
                                for index, (
                                    form,
                                    lemma,
                                    governor,
                                    relation,
                                ) in enumerate(
                                    zip(
                                        batch.FORM[i],
                                        batch.LEMMA[i],
                                        sample_heads,
                                        sample_types,
                                    ),
                                    start=1,
                                )
                            ],
                            metadata=sample_meta,
                        )
                    )

                    tq.update(batch.WORD.size(0))

            with pgzip.open(out, "wt", encoding="utf-8") as fp:
                fp.writelines(sentence.serialize() for sentence in annotated)  # type: ignore

        return True

    def read(self, path: Path) -> list[DataPoint]:
        document: list[TokenList] = [
            handle_contractions(sentence, expand=True)
            for sentence in ConllFileHelper.read(path)
        ]
        data_points: list[DataPoint] = []
        for sentence in document:
            dp = DataPoint(
                word_ids=[self.word_alphabet.get_index(ROOT)],
                char_seq_ids=[[self.char_alphabet.get_index(ROOT_CHAR)]],
                pos_ids=[self.pos_alphabet.get_index(ROOT_POS)],
                forms=[token["form"] for token in sentence],
                lemmata=[token["lemma"] for token in sentence],
                metadata=getattr(sentence, "metadata", None) or {},
            )
            for token in sentence:
                dp.char_seq_ids.append(
                    [
                        self.char_alphabet.get_index(char)
                        for char in token["form"][:MAX_CHAR_LENGTH]
                    ]
                )

                dp.word_ids.append(
                    self.word_alphabet.get_index(DIGIT_RE.sub("0", token["form"]))
                )

                dp.pos_ids.append(
                    self.pos_alphabet.instance2index.get(token["xpos"], 0)
                )
            data_points.append(dp)
        return data_points

    def stack(
        self,
        data: list[DataPoint],
    ) -> StackedData:
        sent_length = MinMax(0)
        char_length = MinMax(0)
        for dp in data:
            sent_length.update(len(dp.word_ids))
            char_length.update(max(len(seq) for seq in dp.char_seq_ids))

        data_size = len(data)
        max_sent_l = sent_length.max
        max_char_l = min(MAX_CHAR_LENGTH, char_length.max)

        wid_inputs = np.full([data_size, max_sent_l], PAD_ID_WORD, dtype=np.int64)
        cid_inputs = np.full(
            [data_size, max_sent_l, max_char_l], PAD_ID_CHAR, dtype=np.int64
        )
        pid_inputs = np.full([data_size, max_sent_l], PAD_ID_TAG, dtype=np.int64)

        masks_e = np.zeros([data_size, max_sent_l], dtype=np.float32)
        lengths = np.empty(data_size, dtype=np.int64)

        forms = []
        lemmata = []
        metadata = []

        for i, (
            wids,
            cid_seqs,
            pids,
            form,
            lemma,
            meta,
        ) in enumerate(data):
            length = len(wids)
            wid_inputs[i, :length] = wids
            for c, cids in enumerate(cid_seqs):
                cid_inputs[i, c, : len(cids)] = cids
            pid_inputs[i, :length] = pids
            masks_e[i, :length] = 1.0
            lengths[i] = length

            forms.append(form)
            lemmata.append(lemma)
            metadata.append(meta)

        return StackedData(
            WORD=wid_inputs,
            CHAR=cid_inputs,
            POS=pid_inputs,
            MASK_ENC=masks_e,
            LENGTH=lengths,
            FORM=forms,
            LEMMA=lemmata,
            META=metadata,
        )


def iterate_batch(
    data: StackedData,
    data_size: int,
    batch_size: int,
) -> Generator[TensorData, None, None]:
    lengths = data.LENGTH

    for idx in range(0, data_size, batch_size):
        batch_slice = slice(idx, idx + batch_size)
        batch_lengths = lengths[batch_slice]
        batch_length = batch_lengths.max().item()
        yield TensorData(
            WORD=torch.from_numpy(data.WORD[batch_slice, :batch_length]),
            CHAR=torch.from_numpy(data.CHAR[batch_slice, :batch_length]),
            POS=torch.from_numpy(data.POS[batch_slice, :batch_length]),
            MASK_ENC=torch.from_numpy(data.MASK_ENC[batch_slice, :batch_length]),
            LENGTH=torch.from_numpy(batch_lengths),
            FORM=data.FORM[batch_slice],
            LEMMA=data.LEMMA[batch_slice],
            META=data.META[batch_slice],
        )
