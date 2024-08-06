import unittest

from dataclasses import dataclass, field
from typing import Self


class BestGuess(int):
    def __str__(self):
        return f"{super().__str__()}"

    def __repr__(self):
        return f"{super().__repr__()}?"

    @classmethod
    def try_parse(cls, value: str, strict=False) -> Self | int | None:
        """
        Raises:
            ValueError: if the value is not a valid integer or an integer with a question mark
        """
        try:
            _value = value.strip()
            if _value.startswith("?") or _value.endswith("?"):
                return cls(_value.strip("?").strip())
            return int(_value)
        except ValueError as e:
            if strict:
                raise ValueError(f"Invalid best guess integer: {value}") from e
            return None


@dataclass
class Author:
    last: str
    first: str | None = field(default=None)
    born: int | None = field(default=None)
    died: int | None = field(default=None)

    @classmethod
    def try_parse(cls, author: str, expected_age: int = 120) -> Self:
        match author.strip().strip(";").strip().split(","):
            case [""]:
                raise ValueError("Empty author")
            case [last]:
                return cls(cls._clean(last))
            case [last, first]:
                return cls(cls._clean(last), cls._clean(first))
            case [last, first, *rest]:
                first = cls._clean(first)
                last = cls._clean(last)
                for other in rest:
                    try:
                        born_s, died_s = other.strip().split("-", maxsplit=1)

                        born = BestGuess.try_parse(born_s, strict=False)
                        died = BestGuess.try_parse(died_s, strict=False)

                        match born, died:
                            case None, None:
                                return cls(
                                    last,
                                    first,
                                )
                            case born, None if born is not None:
                                return cls(
                                    last,
                                    first,
                                    born,
                                    BestGuess(born + expected_age),
                                )
                            case None, died if died is not None:
                                return cls(
                                    last,
                                    first,
                                    BestGuess(died - expected_age),
                                    died,
                                )
                            case born, died:
                                return cls(
                                    last,
                                    first,
                                    born,
                                    died,
                                )
                    except ValueError:
                        pass

                return cls(last, first)
            case _:
                raise ValueError(f"Invalid author: {author}")

    @staticmethod
    def _clean(value: str) -> str:
        """
        Clean an author's name by removing content following opening brackets or parentheses.

        Args:
            value (str): The author's name

        Returns:
            str: The cleaned author's name
        """
        value, *_ = value.split("[", 1)
        value, *_ = value.split("(", 1)
        return value.strip()

    def life_range(
        self,
        round_known: int = 0,
        round_guess: int = 10,
    ) -> tuple[int, int] | None:
        if self.born is None or self.died is None:
            return None

        born = (
            round_int(self.born, round_guess)
            if isinstance(self.born, BestGuess)
            else round_int(self.born, round_known)
        )

        died = (
            round_int(self.died, round_guess)
            if isinstance(self.died, BestGuess)
            else round_int(self.died, round_known)
        )

        return born, died


def round_int(value: int, to: int) -> int:
    match to:
        case None | 0:
            return value
        case _ if not isinstance(to, int):
            raise ValueError(f"Invalid rounding value: not an int {to}")
        case neg if neg < 0:
            val = 10 ** (-neg)
            return int(round(value // val) * val)
        case pos if pos > 0:
            return int(round(value // pos) * pos)
        case _:
            raise ValueError(f"Invalid rounding value: {to}")


class Authors(tuple[Author, ...]):
    @classmethod
    def try_parse(
        cls,
        authors: str | list[str],
        expected_age: int = 120,
        separator: str = ";",
    ) -> Self:
        parsed = []
        authors = authors if isinstance(authors, list) else authors.split(separator)
        for parts in authors:
            parsed.append(Author.try_parse(parts.strip(), expected_age))
        return cls(parsed)


class _TestAuthor(unittest.TestCase):
    def test_base(self):
        self.assertEqual(
            Author.try_parse("Foo, Bar, 1234-5678"),
            Author(last="Foo", first="Bar", born=1234, died=5678),
        )
        self.assertEqual(
            Author.try_parse("Foo, Bar Baz, 1234-5678"),
            Author(last="Foo", first="Bar Baz", born=1234, died=5678),
        )
        self.assertEqual(
            Author.try_parse("Foo, Bar, 1234-5678; "),
            Author(last="Foo", first="Bar", born=1234, died=5678),
        )

    def test_special(self):
        self.assertEqual(
            Author.try_parse("Foo, Bar Baz von [Editor]"),
            Author(last="Foo", first="Bar Baz von", born=None, died=None),
        )
        self.assertEqual(
            Author.try_parse("Foo, Bar Baz von [Editor], 1234-5678"),
            Author(last="Foo", first="Bar Baz von", born=1234, died=5678),
        )
        self.assertEqual(
            Author.try_parse("Foo, B. B. (Bar Baz)"),
            Author(last="Foo", first="B. B.", born=None, died=None),
        )
        self.assertEqual(
            Author.try_parse("Foo, B. B. (Bar Baz), 1234-5678"),
            Author(last="Foo", first="B. B.", born=1234, died=5678),
        )

    def test_missing(self):
        self.assertEqual(
            Author.try_parse("Foo, Bar, 1234-", 100),
            Author(last="Foo", first="Bar", born=1234, died=BestGuess(1334)),
        )
        self.assertEqual(
            Author.try_parse("Foo, Bar, -5678", 100),
            Author(last="Foo", first="Bar", born=5578, died=5678),
        )

    def test_uncertain(self):
        self.assertEqual(
            Author.try_parse("Foo, Bar Baz Bim Bam, 1234-5678?"),
            Author(
                last="Foo",
                first="Bar Baz Bim Bam",
                born=1234,
                died=BestGuess(5678),
            ),
        )
        self.assertEqual(
            Author.try_parse("Foo, Bar Baz Bim Bam, 1234?-5678"),
            Author(
                last="Foo", first="Bar Baz Bim Bam", born=BestGuess(1234), died=5678
            ),
        )
        self.assertEqual(
            Author.try_parse("Foo, Bar Baz Bim Bam, 1234?-5678?"),
            Author(
                last="Foo",
                first="Bar Baz Bim Bam",
                born=BestGuess(1234),
                died=BestGuess(5678),
            ),
        )
        self.assertEqual(
            Author.try_parse("Foo, Bar Baz Bim Bam, 1234?-", 100),
            Author(
                last="Foo",
                first="Bar Baz Bim Bam",
                born=BestGuess(1234),
                died=BestGuess(1334),
            ),
        )
        self.assertEqual(
            Author.try_parse("Foo, Bar Baz Bim Bam, -5678?", 100),
            Author(
                last="Foo",
                first="Bar Baz Bim Bam",
                born=BestGuess(5578),
                died=BestGuess(5678),
            ),
        )

    def test_multiple(self):
        self.assertEqual(
            Authors.try_parse("Foo, Bar, 1234-5678; Bar, Baz, 1-2"),
            (
                Author(last="Foo", first="Bar", born=1234, died=5678),
                Author(last="Bar", first="Baz", born=1, died=2),
            ),
        )


if __name__ == "__main__":
    unittest.main()
