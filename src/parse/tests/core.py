import gzip
import unittest
from abc import abstractmethod
from pathlib import Path

from parse.abc import ParserABC


class TestCohaABC(unittest.TestCase):
    runner: ParserABC

    @abstractmethod
    def setUp(self):
        raise NotImplementedError

    def test_parse(self):
        in_path = Path("../data/conllu/test.coha.conllu.gz")
        out_path = Path(
            f"../data/conllu/test.coha.out.{self.__class__.__name__}.conllu.gz"
        )

        self.assertTrue(self.runner.parse(in_path, out_path))

        with gzip.open(out_path, "rt") as fp:
            print(fp.read())


class TestDtaABC(unittest.TestCase):
    runner: ParserABC

    @abstractmethod
    def setUp(self):
        raise NotImplementedError

    def test_parse(self):
        in_path = Path("../data/conllu/test.dta.conllu.gz")
        out_path = Path(
            f"../data/conllu/test.dta.out.{self.__class__.__name__}.conllu.gz"
        )

        self.assertTrue(self.runner.parse(in_path, out_path))

        with gzip.open(out_path, "rt") as fp:
            print(fp.read())


class IntegrationTestCohaABC(unittest.TestCase):
    runner: ParserABC

    @abstractmethod
    def setUp(self):
        raise NotImplementedError

    def test_parse(self):
        in_path = Path(
            "/storage/corpora/coha/slc/conllu/sampled/text_acad_1820-100000.conllu.gz"
        )
        out_path = Path(
            f"../data/conllu/text_acad_1820.out.{self.__class__.__name__}.conllu.gz"
        )

        self.assertTrue(self.runner.parse(in_path, out_path))

        with gzip.open(out_path, "rt") as fp:
            print("".join(fp.readlines()[:100]))
