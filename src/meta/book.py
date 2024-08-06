from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import unittest
from defusedxml import ElementTree

import requests

from meta.author import Author, Authors


@dataclass
class BookMeta:
    year: int
    title: str | None = field(default=None)
    authors: Authors = field(default_factory=tuple)
    extra: dict[str, str] = field(default_factory=dict)


class BookMetaSearch(ABC):
    @classmethod
    @abstractmethod
    def query(cls, title: str, *args, **kwargs) -> BookMeta | None:
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def _process_response(cls, response: requests.Response) -> BookMeta | None:
        raise NotImplementedError


class OpenLibrary(BookMetaSearch):
    @classmethod
    def query(cls, title: str, authors: Authors = None, **kwargs):
        query = f"title:{title}"

        life_range = None
        born, died = 9999, -9999
        for author in authors or ():
            if author.born or author.died:
                born_died = author.life_range(2, 10)
                if born_died:
                    born, died = min(born, born_died[0]), max(died, born_died[1])
                    life_range = range(born, died)

        if life_range:
            pass
            # query = f"{query} author:{authors[0].last}"
            # query = f"{query} first_publish_year:[{born} TO {died}]"

        params = {
            # "q": query,
            "title": title,
            "fields": "author_key,author_name,first_publish_year,publish_date,title",
            # "fields": "author_key,author_name,author_alternative_name,first_publish_year,publish_date,title,title_suggest,title_sort",
            # "sort": "old",
            "limit": "100",
        }
        response = requests.get(
            "https://openlibrary.org/search.json",
            params=params | kwargs,
            timeout=30,
        )

        return cls._process_response(response, life_range)

    @classmethod
    def _process_response(
        cls,
        response: requests.Response,
        life_range: range | None = None,
    ):
        try:
            response_json = response.json()
            docs = response_json.get("docs", None)
            if docs:
                for doc in docs:
                    author_name = doc.get("author_name", None)
                    year = doc.get("first_publish_year")
                    if year is None:
                        publish_date: list[str] = doc.get("publish_date")
                        if publish_date:
                            year = min(
                                int(y.strip())
                                for y in publish_date
                                if y.strip().isnumeric()
                            )
                    else:
                        year = int(year)

                    if life_range and year not in life_range:
                        continue

                    return BookMeta(
                        year=year,
                        title=doc["title"],
                        authors=author_name and Authors.try_parse(author_name),
                        extra=doc,
                    )
        except requests.exceptions.JSONDecodeError:
            return None


ns = {
    "rdf": "http://www.w3.org/1999/02/22-rdf-syntax-ns#",
    "dc": "http://purl.org/dc/elements/1.1/",
    "dcterms": "http://purl.org/dc/terms/",
    "gndo": "https://d-nb.info/standards/elementset/gnd#",
}


class DNB(BookMetaSearch):
    @classmethod
    def query(
        cls,
        title,
        authors: Authors | None = None,
        sortby="sortby jhr/sort.ascending",
        **kwargs,
    ):
        query = f'dc.title="{title}"'

        life_range = None
        if authors:
            for author in authors:
                born_died = author.life_range(2, 10)
                if born_died:
                    born, died = born_died
                    life_range = range(born, died)
                    # query = f"{query} and dnb.jhr within {born}-{died}"
                    break

        if sortby:
            query = " ".join((query, sortby))

        params = {
            "version": "1.1",
            "operation": "searchRetrieve",
            "query": query,
            "maximumRecords": "100",
        } | kwargs

        response = requests.get(
            "http://services.dnb.de/sru/dnb",
            params=params,
            timeout=30,
        )
        return cls._process_response(response, life_range)

    @classmethod
    def _process_response(
        cls,
        response: requests.Response,
        life_range: range | None = None,
    ):
        try:
            root = ElementTree.fromstring(response.text)
            for record in root.findall(".//rdf:Description", namespaces=ns):
                year = record.find("dcterms:issued", namespaces=ns)
                if year is None or year.text is None:
                    continue

                year = int(year.text)
                if life_range and year not in life_range:
                    continue

                title = record.find("dc:title", namespaces=ns)
                title = title.text if title is not None else None
                title = title.strip() if title is not None else None
                title = title or None

                creator = record.find("dcterms:creator", namespaces=ns)
                creator = creator.text if creator is not None else None
                creator = creator.strip() if creator is not None else None
                creator = creator or None

                return BookMeta(year, title, creator)
            return None
        except ValueError:
            return None
        except ElementTree.ParseError:
            return None


class _TestOpenLibrary(unittest.TestCase):
    def test_ol_iphigenie(self):
        meta = OpenLibrary.query("Iphigenie auf Tauris")
        self.assertIsNotNone(meta)

        if meta:
            print(meta)
            self.assertEqual(1787, meta.year)

    def test_ol_iphigenie_goethe(self):
        meta = OpenLibrary.query(
            "Iphigenie auf Tauris",
            (Author("Goethe", "Johann Wolfgang von", 1749, 1832),),
        )
        self.assertIsNotNone(meta)

        if meta:
            print(meta)
            self.assertEqual(1787, meta.year)

    def test_ol_faust(self):
        meta = OpenLibrary.query("Faust")
        self.assertIsNotNone(meta)

        if meta:
            print(meta)
            self.assertEqual(1800, meta.year, "expected wrong year 1800, actually 1808")

    def test_ol_faust_goethe(self):
        meta = OpenLibrary.query(
            "Faust",
            (Author("Goethe", "Johann Wolfgang von", 1749, 1832),),
        )
        self.assertIsNotNone(meta)

        if meta:
            print(meta)
            self.assertEqual(1800, meta.year, "expected wrong year 1800, actually 1808")

    def test_ol_gwissenswurm(self):
        meta = OpenLibrary.query("Der G'wissenswurm")
        self.assertIsNotNone(meta)

        if meta:
            print(meta)
            self.assertEqual(meta.year, 1964, "expected wrong year 1964, actually 1874")

    # def test_ol_gwissenswurm_subtitle(self):
    #     meta = OpenLibrary.query("Der G'wissenswurm: Bauernkomödie in drei Akten")
    #     self.assertIsNotNone(meta)
    #     self.assertEqual(2017, meta.year)  # this is wrong, but expected

    def test_ol_buddenbrooks(self):
        meta = OpenLibrary.query("Buddenbrooks")
        self.assertIsNotNone(meta)

        if meta:
            print(meta)
            self.assertEqual(1909, meta.year)

    def test_ol_declaration(self):
        meta = OpenLibrary.query(
            "The Declaration of Independence of the United States of America"
        )
        self.assertIsNotNone(meta)

        if meta:
            print(meta)
            self.assertEqual(1776, meta.year)

    def test_ol_declaration_jefferson(self):
        meta = OpenLibrary.query(
            "The Declaration of Independence of the United States of America",
            Author("Jefferson", "Thomas", 1743, 1826),
        )
        self.assertIsNotNone(meta)

        if meta:
            print(meta)
            self.assertEqual(1776, meta.year)


class _TestDNB(unittest.TestCase):
    def test_dnb_faust(self):
        meta = DNB.query("Faust")
        self.assertIsNotNone(meta)

        if meta:
            print(meta)
            self.assertEqual(1691, meta.year, "expected wrong year 1691 actually 1808")

    def test_dnb_faust_goethe(self):
        meta = DNB.query(
            "Faust",
            (Author("Goethe", "Johann Wolfgang von", 1749, 1832),),
        )
        self.assertIsNotNone(meta)

        if meta:
            print(meta)
            self.assertEqual(1790, meta.year, "expected wrong year 1790 actually 1808")

    def test_dnb_iphigenie(self):
        meta = DNB.query("Iphigenie auf Tauris")
        self.assertIsNotNone(meta)

        if meta:
            print(meta)
            self.assertEqual(1787, meta.year)

    def test_dnb_gwissenswurm(self):
        meta = DNB.query("Der G'wissenswurm")
        self.assertIsNotNone(meta)

        if meta:
            print(meta)
            self.assertEqual(1874, meta.year)

    def test_dnb_gwissenswurm_subtitle(self):
        meta = DNB.query("Der G'wissenswurm: Bauernkomödie in drei Akten")
        self.assertIsNotNone(meta)

        if meta:
            print(meta)
            self.assertEqual(2017, meta.year, "expected wrong year 2017 actually 1874")

    def test_dnb_buddenbrooks(self):
        meta = DNB.query("Buddenbrooks")
        self.assertIsNotNone(meta)

        if meta:
            print(meta)
            self.assertEqual(1909, meta.year)


if __name__ == "__main__":
    unittest.main()
