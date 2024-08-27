import json
from pathlib import Path
from typing import Final, Generator, Literal, NamedTuple

import numpy as np
import torch
from attr import dataclass
from conllu import Metadata
from stanza.models.common.doc import Document as StanzaDocument
from stanza.models.common.doc import Sentence as StanzaSentence
from stanza.models.common.doc import Word as StanzaWord
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
from parse.stanza_ import ParserWithStanzaPreProcessor
from parse.utils import ConllFileHelper

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
    sentence: StanzaSentence


@dataclass
class StackedData:
    word_ids: np.ndarray
    char_seq_ids: np.ndarray
    pos_ids: np.ndarray
    mask_enc: np.ndarray
    lengths: list[int]
    sentences: list[StanzaSentence]

    def __getitem__(self, key) -> np.ndarray | list[int] | list[StanzaSentence]:
        if not hasattr(self, key):
            raise KeyError(key)
        return getattr(self, key)


@dataclass
class TensorData:
    word_ids: torch.Tensor
    char_seq_ids: torch.Tensor
    pos_ids: torch.Tensor
    mask_enc: torch.Tensor
    lengths: list[int]
    sentences: list[StanzaSentence]

    def __getitem__(self, key) -> torch.Tensor | list[Metadata]:
        if not hasattr(self, key):
            raise KeyError(key)
        return getattr(self, key)

    def to(self, device: str | torch.device) -> "TensorData":
        self.word_ids = self.word_ids.to(device)
        self.char_seq_ids = self.char_seq_ids.to(device)
        self.pos_ids = self.pos_ids.to(device)
        self.mask_enc = self.mask_enc.to(device)
        return self


class StackPointerParser(ParserWithStanzaPreProcessor):
    def __init__(
        self,
        language: Literal["en", "de"] = "en",
        batch_size: int = 128,
        beam: int = 1,
        base_path: Path | None = None,
        quiet: bool = False,
        preprocess: bool = True,
        device: str | torch.device = "cpu",
        **kwargs,
    ):
        kwargs = {
            "processors": "tokenize,mwt,pos,lemma",
            "preprocess_device": device,
            "batch_size": kwargs.pop("pos_batch_size", 1024),
        } | kwargs

        super().__init__(
            language=language,
            preprocess=preprocess,
            device=device,
            **kwargs,
        )

        self.batch_size = batch_size
        self.beam = beam
        self.quiet = quiet
        base_path = Path(base_path or BASE_PATH)

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

    def process(self, path: Path, out: Path):
        document, filtered = self.pre_process(path)
        datapoints = self.prepare(filtered)

        with tqdm(
            total=len(datapoints),
            desc=f"Parsing: {path.name}",
            position=1,
            leave=False,
            ascii=True,
            smoothing=0,
            disable=self.quiet,
        ) as tq:
            for batch in iterate_batch(
                self.stack(datapoints),
                len(datapoints),
                self.batch_size,
            ):
                with torch.no_grad():
                    batch_d = batch.to(self.device)
                    heads, types = self.nlp.decode(
                        batch_d.word_ids,
                        batch_d.char_seq_ids,
                        batch_d.pos_ids,
                        mask=batch_d.mask_enc,
                        beam=self.beam,
                    )

                for i, (length, sentence) in enumerate(
                    zip(batch.lengths, batch.sentences)
                ):
                    sample_heads = heads[i][1 : length + 1]
                    sample_types = types[i][1 : length + 1]

                    for word, governor, relation in zip(
                        sentence.words, sample_heads, sample_types
                    ):
                        word: StanzaWord
                        word.head = governor
                        word.deprel = self.type_alphabet.get_instance(relation)

                    tq.update(batch.word_ids.size(0))

        if self.drop_filtered:
            self.write(filtered, out)
        else:
            self.write(document, out)

        return path

    def prepare(self, filtered: StanzaDocument) -> list[DataPoint]:
        data_points: list[DataPoint] = []
        for sentence in filtered.sentences:
            dp = DataPoint(
                word_ids=[self.word_alphabet.get_index(ROOT)],
                char_seq_ids=[[self.char_alphabet.get_index(ROOT_CHAR)]],
                pos_ids=[self.pos_alphabet.get_index(ROOT_POS)],
                sentence=sentence,
            )

            for word in sentence.words:
                dp.char_seq_ids.append(
                    [
                        self.char_alphabet.get_index(char)
                        for char in word.text[:MAX_CHAR_LENGTH]
                    ]
                )

                dp.word_ids.append(
                    self.word_alphabet.get_index(DIGIT_RE.sub("0", word.text))
                )

                dp.pos_ids.append(self.pos_alphabet.instance2index.get(word.xpos, 0))
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

        lengths = []
        sentences = []

        for i, (
            wids,
            cid_seqs,
            pids,
            sentence,
        ) in enumerate(data):
            length = len(wids)
            wid_inputs[i, :length] = wids
            for c, cids in enumerate(cid_seqs):
                cid_inputs[i, c, : len(cids)] = cids
            pid_inputs[i, :length] = pids
            masks_e[i, :length] = 1.0

            lengths.append(length)
            sentences.append(sentence)

        return StackedData(
            word_ids=wid_inputs,
            char_seq_ids=cid_inputs,
            pos_ids=pid_inputs,
            mask_enc=masks_e,
            lengths=lengths,
            sentences=sentences,
        )


def iterate_batch(
    data: StackedData,
    data_size: int,
    batch_size: int,
) -> Generator[TensorData, None, None]:
    lengths = data.lengths

    for idx in range(0, data_size, batch_size):
        batch_slice = slice(idx, idx + batch_size)
        batch_lengths = lengths[batch_slice]
        batch_length = max(batch_lengths)
        yield TensorData(
            word_ids=torch.from_numpy(data.word_ids[batch_slice, :batch_length]),
            char_seq_ids=torch.from_numpy(
                data.char_seq_ids[batch_slice, :batch_length]
            ),
            pos_ids=torch.from_numpy(data.pos_ids[batch_slice, :batch_length]),
            mask_enc=torch.from_numpy(data.mask_enc[batch_slice, :batch_length]),
            lengths=batch_lengths,
            sentences=data.sentences[batch_slice],
        )
