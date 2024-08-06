from pathlib import Path
from typing import Literal

import pgzip
import torch
from stanza import Pipeline
from stanza.models.common.doc import Document
from stanza.pipeline.core import DownloadMethod
from stanza.utils.conll import CoNLL
from tqdm import tqdm

from parse.abc import ParserABC
from parse.utils import ConllFileHelper, batched, tokenlist_to_str


class StanzaRunner(ParserABC):
    def __init__(
        self,
        language: Literal["en", "de"] = "en",
        device: str | torch.device = "cpu",
        batch_size: int = 128,
        quiet: bool = False,
    ):
        device = torch.device(device)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise ValueError("device.type == 'cuda', but no GPU is available")

        self.batch_size = batch_size
        self.nlp = Pipeline(
            lang=language,
            processors="tokenize,mwt,pos,lemma,depparse",
            tokenize_no_ssplit=True,
            download_method=DownloadMethod.REUSE_RESOURCES,
            device=device,
            depparse_batch_size=batch_size,
        )
        self.quiet = quiet

    def parse(self, path: Path, out: Path):
        out.parent.mkdir(parents=True, exist_ok=True)

        sentences = ConllFileHelper.read(path)

        with pgzip.open(out, "wt", encoding="utf-8") as fp:
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
                    text: str = "\n\n".join(
                        tokenlist_to_str(sentence, expand_contractions=True).replace(
                            "ſ", "s"
                        )
                        for sentence in batch
                    )
                    document: Document = self.nlp(text)

                    for stanza_sent, orig_sent in zip(document.sentences, sentences):
                        for key, value in orig_sent.metadata.items():
                            if key == "text":
                                continue
                            stanza_sent.add_comment(f"# {key} = {value}")

                    CoNLL.write_doc2conll(document, fp, mode="a")

                    tq.update(len(batch))

        return True
