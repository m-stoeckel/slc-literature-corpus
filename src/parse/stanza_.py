import logging
from pathlib import Path
from typing import Any, Literal, Optional

import torch
from conllu.models import TokenList
from stanza import Pipeline
from stanza.models.common.doc import Document, Sentence
from stanza.pipeline.core import DownloadMethod

from parse.abc import ParserABC
from parse.utils import ConllFileHelper


def copy_metadata(conllu_sentence: TokenList, stanza_sentence: Sentence) -> Sentence:
    for key, value in conllu_sentence.metadata.items():
        if key == "text":
            continue
        stanza_sentence.add_comment(f"# {key} = {value}")
    return stanza_sentence


def copy_metadata_(sentence_tuple: tuple[TokenList, Sentence]) -> Sentence:
    return copy_metadata(*sentence_tuple)


def get_metadata_from_stanza(sentence: Sentence) -> dict[str, str]:
    metadata = {}
    for comment in sentence.comments:
        key, value = comment.strip("#").split("=", 1)
        metadata[key.strip()] = value.strip()
    return metadata


class StanzaParser(ParserABC):
    def __init__(
        self,
        language: Literal["en", "de"] = "en",
        processors="tokenize,mwt,pos,lemma,depparse",
        device: str | torch.device = "cpu",
        batch_size: int = 1024,
        **kwargs,
    ):
        """
        Initialize a StanzaRunner instance. This class is a wrapper around the Stanza library.

        Args:
            language (Literal[&quot;en&quot;, &quot;de&quot;], optional): The annotation language. Defaults to "en".
            processors (str, optional): The processors to use. Defaults to "tokenize,mwt,pos,lemma,depparse".
            device (str | torch.device, optional): The device to run the pipeline on. Defaults to "cpu".
            batch_size (int, optional): The batch size for POS and dependency parsing. Defaults to 1024.
            min_sentence_len (int, optional): The minimum sentence length to retain. Set to -1 (default) to disable.
            max_sentence_len (int, optional): The maximum sentence length to retain. Set to -1 (default) to disable.
            drop_filtered (bool, optional): Whether to drop the filtered sentences. Defaults to False.
        """
        super().__init__(language=language, device=device, **kwargs)

        if not processors.startswith("tokenize,mwt"):
            if "tokenize" not in processors:
                processors = "tokenize,mwt," + processors
                logging.warning(
                    f"Prepending 'tokenize,mwt,' processors to: {processors}"
                )
            else:
                raise ValueError(
                    f"Expected processors to start with 'tokenize,mwt', got: {processors}"
                )
        self.processors = processors
        self.processors_pre = "tokenize,mwt"
        self.processors_post = (
            processors.removeprefix("tokenize,mwt").strip().strip(",")
        )

        self.pipeline = Pipeline(
            lang=language,
            processors=processors,
            tokenize_no_ssplit=True,
            download_method=DownloadMethod.REUSE_RESOURCES,
            device=device,
            pos_batch_size=batch_size,
            depparse_batch_size=batch_size,
        )

    @staticmethod
    def read(path: Path) -> Document:
        return ConllFileHelper.read_stanza(path)

    @staticmethod
    def write(document: Document, out: Path) -> None:
        out.parent.mkdir(parents=True, exist_ok=True)
        ConllFileHelper.write_stanza(document, out)

    def parse(self, document, processors: str | None = None) -> Document:
        return self.pipeline(document, processors=processors or self.processors)

    def process(self, path: Path | str, out: Path | str):
        path, out = Path(path), Path(out)

        document, filtered = self.pre_process(path)

        if self.processors_post:
            self.pipeline(filtered, processors=self.processors_post)

        if self.drop_filtered:
            self.write(filtered, out)
        else:
            self.write(document, out)

        return path

    def pre_process(self, path: Path) -> tuple[Document, Document]:
        original: Document = self.read(path)
        document: Document = self.pipeline(original, processors=self.processors_pre)

        for doc_sentence, orig_sentence in zip(document.sentences, original.sentences):
            doc_sentence._comments = orig_sentence._comments

        filtered: Document = self.filter(document)

        return document, filtered

    def filter(
        self,
        document: Document,
    ) -> Document:
        """
        Filter the sentences in a document based on their length.
        If neither `min_sentence_len` nor `max_sentence_len` are set, the original document is returned.
        To retain references to the original sentences, all subsequent processing must NOT contain 'tokenize,mwt' processors.

        Args:
            document (Document): The document to filter.

        Returns:
            Document: A new `Document` object containing references to the filtered sentences.
        """
        if self.min_sentence_len < 0 and self.max_sentence_len < 0:
            return document

        filtered = Document([])
        for sentence in document.sentences:
            if (
                self.min_sentence_len > 0
                and len(sentence.words) >= self.min_sentence_len
            ) or (
                self.max_sentence_len > 0
                and len(sentence.words) <= self.max_sentence_len
            ):
                filtered.sentences.append(sentence)
        filtered._count_words()
        return filtered


class ParserWithStanzaPreProcessor(StanzaParser):
    pipeline: Optional[Pipeline] = None

    def __init__(
        self,
        language: Literal["en", "de"] = "en",
        preprocess: bool = True,
        processors: str = "tokenize,mwt",
        preprocess_device: str | torch.device = "cpu",
        device: str | torch.device = "cpu",
        **kwargs,
    ):

        self.do_preprocess = preprocess
        if not self.do_preprocess:
            ParserABC.__init__(self, language=language, device=device, **kwargs)
        else:
            super().__init__(
                language=language,
                processors=processors,
                device=preprocess_device,
                **kwargs,
            )
            self.__setattr__ = self.__setattr_safe__

    def pre_process(self, path: Path) -> tuple[Document, Document]:
        if not self.do_preprocess or self.pipeline is None:
            original: Document = self.read(path)
            filtered: Document = self.filter(original)
            return original, filtered

        document, filtered = super().pre_process(path)

        if self.processors_post:
            self.pipeline(filtered, processors=self.processors_post)

        return document, filtered

    def __setattr_safe__(self, name: str, value: Any) -> None:
        if name == "pipeline":
            if self.do_preprocess and (
                not hasattr(self, "pipeline")
                or object.__getattribute__(self, "pipeline") is not None
            ):
                raise AttributeError(
                    f"Cannot set pipeline attribute of a {type(self).__name__} instance when preprocessing is enabled! "
                    f"This {type(self).__name__}'s pipeline attribute is currently bound to: {self.pipeline}."
                )
            return object.__setattr__(self, name, value)
        return super().__setattr__(name, value)
