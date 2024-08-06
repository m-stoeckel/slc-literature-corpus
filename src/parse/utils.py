import random
import re
from argparse import ArgumentParser
from collections import defaultdict
from dataclasses import dataclass
from itertools import islice
from pathlib import Path
from typing import Callable, Final, Generator, Iterable, Iterator, TypeVar

import pgzip
from conllu import parse
from conllu.models import SentenceList, Token, TokenList
from pqdm.processes import pqdm
from tqdm import tqdm


def get_parser(pattern: str):
    parser = ArgumentParser()
    parser.add_argument(
        "in_root",
        type=Path,
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default=pattern,
    )

    parser.add_argument(
        "out_root",
        type=Path,
    )

    parser.add_argument(
        "-l",
        "--language",
        type=str,
        choices=["en", "de"],
        default="en",
    )
    parser.add_argument(
        "--en",
        action="store_const",
        const="en",
        dest="language",
    )
    parser.add_argument(
        "--de",
        action="store_const",
        const="de",
        dest="language",
    )

    parser.add_argument(
        "-n",
        "-ng",
        "--num_gpu",
        type=int,
        default=None,
    )
    parser.add_argument(
        "-nc",
        "--num_cpu",
        type=int,
        default=None,
    )

    parser.add_argument(
        "-sg",
        "--scale_gpu",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "-sc",
        "--scale_cpu",
        type=float,
        default=1.0,
    )

    return parser


@dataclass
class IsStandalone:
    first_is_upper: bool | None = None
    last_is_punctuation: bool | None = None
    has_verb: bool | None = None
    even_quotes: bool | None = None
    balanced_brackets: bool | None = None

    def is_standalone(self) -> bool:
        return all(
            (
                self.first_is_upper,
                self.last_is_punctuation,
                self.has_verb,
                self.even_quotes,
                self.balanced_brackets,
            )
        )


EOS_MARKERS: Final[set[str]] = set(".?!")


@dataclass
class SentenceValidator:
    pos_verb: set[str]

    def check(self, text: str, pos: set[str]) -> IsStandalone:
        """
        Check if a sentence is a standalone sentence.
        Standalone sentences must meet the following requirements:
        - Sentences must start with a capitalized character.
        - Sentences must end with a period, or a question mark, or an exclamation mark.
        - Sentences must contain a verb based on the part-of-speech tags.
        - The number of (double) quotation marks must be even.
        - The number of left brackets must be equal to that of right brackets.

        Args:
            sent (T_co): The sentence to be checked.

        Returns:
            Self: An instance of the SentenceValidatorABC class.
        """
        if not text.strip()[0].isupper():
            return IsStandalone(False)
        if text.strip()[-1] not in EOS_MARKERS:
            return IsStandalone(True, False)
        if not any(p in pos for p in self.pos_verb):
            return IsStandalone(True, True, False)
        if text.count('"') % 2 != 0:
            return IsStandalone(True, True, True, False)
        if text.count("(") != text.count(")"):
            return IsStandalone(True, True, True, True, False)
        return IsStandalone(True, True, True, True, True)


class ConllFileHelper:
    @classmethod
    def read(cls, path: Path) -> SentenceList:
        with pgzip.open(path, "rt", encoding="utf-8") as fp:
            return parse(fp.read())  # type: ignore

    @classmethod
    def read_list(cls, path: Path) -> list:
        return [list(sentence) for sentence in cls.read(path)]


def conll_space_after_(token: Token | dict) -> str:
    form: str = token["form"]
    misc = token.get("misc", {})
    if misc is None or "SpaceAfter" not in misc or misc["SpaceAfter"] != "No":
        return form + " "
    return form


def handle_contractions(sentence: TokenList, expand=True) -> TokenList:
    """
    Expand contractions in a sentence or skip the expanded tokens.

    Args:
        sentence (TokenList): The sentence to process.
        expand (bool, optional): If True, contractions will be expanded.
            Otherwise, the expanded forms will be skipped.
            Defaults to False.

    Returns:
        TokenList: The processed sentence.
    """
    tokens = []
    skip = 0
    for token in sentence:
        if skip > 0:
            skip -= 1
            continue

        if not isinstance(token["id"], int):
            # skip the contracted token if expand is True
            if expand:
                continue

            # or skip the following tokens, i.e. the expanded contraction
            # contractions are denoted as id ranges (like 4-5) in the id field
            a, _, b = token["id"]

            # so we calculate the number of tokens to skip be the difference + 1
            skip = (b - a) + 1
        tokens.append(token)
    tokenlist = TokenList(tokens)
    tokenlist.metadata = sentence.metadata
    return tokenlist


def tokenlist_to_str(
    sentence: TokenList,
    tokenized=False,
    expand_contractions=True,
) -> str:
    """
    Convert a TokenList to a string.

    Args:
        sentence (TokenList): The sentence to convert.
        tokenized (bool, optional): If True, tokens will be separated by a space.
            Otherwise, SpaceAfter=No tokens will be concatenated without a space.
            Defaults to False.
        expand_contractions (bool, optional): If True, contractions will be expanded.
            Otherwise, the expanded forms will be skipped.
            Defaults to False.

    Returns:
        str: The sentence as a string.
    """
    sentence = handle_contractions(sentence, expand=expand_contractions)
    if tokenized:
        return " ".join(token["form"] for token in sentence)
    return "".join(conll_space_after_(token) for token in sentence)


def sample_from_conllu(
    folder: Path,
    out: Path,
    k: int = 450,
    pattern: str = "*.conllu.gz",
    seed: int = 42,
    lengths: list[int] | None = None,
    period: int = 3,
    group_by_regex: str | None = None,
    group_by_callback: Callable[[dict[str, str]], str] | None = None,
    group_all_match: bool = True,
):
    lengths = lengths or [5, 10, 15, 20, 30, 40, 50, 60, 70]
    length_ranges = [range(length, length + period) for length in lengths]

    files = list(sorted(folder.glob(pattern)))

    if group_by_regex is not None:
        group_by_regex_pattern: re.Pattern = re.compile(group_by_regex)
        grouped = defaultdict(list)
        for file in files:
            match = group_by_regex_pattern.search(str(file.absolute()))

            if match is None:
                if group_all_match:
                    raise ValueError(
                        f"File {file} does not match the group pattern: {group_by_regex_pattern.pattern}"
                    )
                print(
                    f"Warning: File {file} does not match the group pattern: {group_by_regex_pattern.pattern}"
                )
                continue

            if group_by_callback is not None:
                key = group_by_callback(match.groupdict(""))
            else:
                key = "".join(value for _, value in sorted(match.groupdict("").items()))

            grouped[key].append(file)
    else:
        grouped = {folder.name: files}

    for name, group in tqdm(
        sorted(grouped.items()),
        desc="Grouping",
        position=1,
        leave=False,
    ):
        group: list[Path] = list(sorted(group))

        documents: list[SentenceList] = []
        iterators: list[Iterator[int]] = []
        file_names: list[str] = []

        for file in tqdm(group, desc="Loading", position=2, leave=False):
            file_names.append(file.stem)

            document = ConllFileHelper.read(file)
            documents.append(document)

            random.seed(seed)
            indices = list(range(len(document)))
            random.shuffle(indices)
            iterators.append(iter(indices))

        counter = 0
        per_length_counter = {length.start: 0 for length in length_ranges}
        sentences: list[TokenList] = []
        with tqdm(total=k, desc="Processing", position=2, leave=False) as tq:
            while (
                any(length < k for length in per_length_counter.values())
                and len(iterators) > 0
            ):
                to_remove = []
                for i in range(len(iterators)):
                    file_name: str = file_names[i]
                    it: Iterator[int] = iterators[i]
                    document: SentenceList = documents[i]
                    try:
                        index = next(it)
                        sentence: TokenList = document[index]

                        length_range = None
                        for range_ in length_ranges:
                            if len(sentence) in range_:
                                length_range = range_
                                break
                        if length_range is None:
                            continue

                        per_length_counter[length_range.start] += 1
                        if per_length_counter[length_range.start] > k:
                            continue

                        # update metadata with doc_id
                        sentence.metadata["doc_id"] = sentence.metadata.get(
                            "doc_id", file_name
                        )
                        # preserve original sent_id in orig_id
                        # or set it to the (1-indexed) offset in the document
                        sentence.metadata["orig_id"] = sentence.metadata.get(
                            "orig_id", sentence.metadata.get("sent_id", index + 1)
                        )

                        sentence.metadata["sent_id"] = (counter := counter + 1)
                        sentence.metadata["text"] = sentence.metadata.get(
                            "text",
                            tokenlist_to_str(sentence, expand_contractions=False),
                        )

                        sentences.append(sentence)
                        tq.update(1)
                    except StopIteration:
                        to_remove.append(i)

                # remove iterators that are exhausted
                # start with the highest index to avoid shifting
                for i in reversed(to_remove):
                    del iterators[i]
                    del file_names[i]
                    del documents[i]

            tq.total = len(sentences)

        out_path = out / (name + f"-{k}.conllu.gz")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with pgzip.open(out_path, "wt") as out_file:
            out_file.writelines(sentence.serialize() for sentence in sentences)  # type: ignore


T = TypeVar("T")


def batched(iterable: Iterable[T], n: int) -> Generator[tuple[T, ...], None, None]:
    """Batch data into tuples of length n. The last batch may be shorter."""
    # batched('ABCDEFG', 3) --> ABC DEF G
    if n < 1:
        raise ValueError("n must be at least one")
    it = iter(iterable)
    while batch := tuple(islice(it, n)):
        yield batch


def batched_lists(
    iterable: Iterable[list[T]],
    n: int,
    drop=False,
) -> Generator[tuple[list[T], ...], None, None]:
    """
    Yield batches of lists up to a total of n total items per batch.
    Lists that are longer than n items are dropped silently if drop is True. Otherwise, they are yielded as a single batch.

    Args:
        iterable (Iterable[list[T]]): Lists to batch.
        n (int): The maximum number of items per batch.
        drop (bool, optional): Whether to drop lists that are longer than n. Defaults to False.
    """
    if n < 1:
        raise ValueError("n must be at least one")
    batch = []
    for sentence in iterable:
        if len(batch) + len(sentence) > n:
            yield tuple(batch)
            batch.clear()
        if len(sentence) > n:
            if drop:
                continue
            yield (sentence,)
            continue
        batch.append(sentence)


EMPTY_CONLLU_TOKEN = {
    key: "_"
    for key in (
        "id",
        "form",
        "lemma",
        "upos",
        "xpos",
        "feats",
        "head",
        "deprel",
        "deps",
        "misc",
    )
}
