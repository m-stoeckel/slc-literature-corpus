import bz2
import itertools
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Generator, Literal
from xml.etree import ElementTree as ET

import pgzip
import spacy
from spacy.tokens import Doc
from tqdm import tqdm

from parse.abc import ParserABC
from parse.utils import SentenceValidator

NS: Final[dict[str, str]] = {
    "cas": "http:///uima/cas.ecore",
    "dkpro.metadata": "http:///de/tudarmstadt/ukp/dkpro/core/api/metadata/type.ecore",
    "dkpro.structure": "http:///de/tudarmstadt/ukp/dkpro/core/api/structure/type.ecore",
    "tcas": "http:///uima/tcas.ecore",
    "xmi": "http://www.omg.org/XMI",
}


@dataclass
class Paragraph:
    begin: int
    end: int


@dataclass
class File:
    sofaString: str
    paragraphs: list[Paragraph]

    def __iter__(self) -> Generator[str, None, None]:
        for paragraph in self.paragraphs:
            if not paragraph.begin == paragraph.end:
                yield self.sofaString[paragraph.begin : paragraph.end]

    def __len__(self) -> int:
        return len(self.paragraphs)

    @classmethod
    def from_xml(cls, path: Path):
        tree = ET.parse(bz2.open(path, "rt", encoding="utf-8"))

        sofa = tree.find("cas:Sofa", NS)
        if sofa is None:
            raise ValueError(f"No sofa found in XML file: {path}")

        sofaString = sofa.attrib["sofaString"]
        paragraphs = list(tree.findall("dkpro.structure:Paragraph", NS))

        return cls(
            sofaString=sofaString,
            paragraphs=[
                Paragraph(
                    begin=int(paragraph.attrib["begin"]),
                    end=int(paragraph.attrib["end"]),
                )
                for paragraph in paragraphs
            ],
        )


class SpacyRunner(ParserABC):
    def __init__(
        self,
        language: Literal["en_core_web_sm", "de_core_news_sm"] = "en_core_web_sm",
        parser: bool = False,
        validate: bool = True,
        quiet: bool = False,
    ):
        # enable = ["tagger", "attribute_ruler"]
        # exclude = ["ner", "tok2vec", "senter"]

        if parser:
            exclude = ["ner"]
            # enable.extend(["parser", "lemmatizer"])
        else:
            exclude = ["ner", "parser"]
            # exclude.extend(["parser", "lemmatizer"])

        spacy.prefer_gpu()
        self.nlp = spacy.load(language, exclude=exclude)
        self.nlp.add_pipe("sentencizer")
        self.nlp.add_pipe("conll_formatter", last=True)

        self.validate = validate
        self.quiet = quiet

    def parse(self, path: Path, out: Path):
        out.parent.mkdir(parents=True, exist_ok=True)

        document = File.from_xml(path)

        conll = []
        for paragraph in tqdm(
            document,
            desc=f"Parsing: {path.name}",
            position=1,
            leave=False,
            ascii=True,
            smoothing=0,
            disable=self.quiet,
        ):
            conll.extend(self.run_paragraph(paragraph))

        self.write(out, "\n\n".join(conll))

        return True

    def write(self, out: Path, content: str):
        with pgzip.open(out, "wt", encoding="utf-8") as fp:
            fp.write(content)  # type: ignore

    def run_paragraph(self, paragraph: str) -> Generator[str, None, None]:
        paragraph = paragraph.replace("\n", " ")

        doc: Doc = self.nlp(paragraph)

        # Add a None to the end of the iterator to include the last sentence if it ends with a semicolon.
        it = itertools.chain(doc.sents, (None,))

        semicolon_begin = None
        validator = SentenceValidator({"VERB", "AUX"})
        for sentence in it:
            if sentence and sentence.text.strip().endswith(";"):
                semicolon_begin = semicolon_begin or sentence.start
                continue

            # If the previous sentence(s) ends with a semicolon, then the current sentence is a continuation of the previous sentence(s).
            if semicolon_begin is not None:
                # If the very last sentence ended with a semicolon, we include it anyways.
                sentence_end = sentence and sentence.end
                sentence = doc[semicolon_begin:sentence_end]

            if sentence is None:
                continue

            if (
                not self.validate
                or validator.check(
                    sentence.text,
                    {tok.pos_ for tok in sentence},
                ).is_standalone()
            ):
                conll_string: str = sentence._.conll_str
                if conll_string is not None:
                    yield conll_string.strip()
