from queue import Empty, PriorityQueue, Queue
from threading import ThreadError
from typing import Literal

from stanza.server import CoreNLPClient, StartServer

from parse.abc import ParserABC


class CoreNlpServer(CoreNLPClient):
    def __init__(
        self,
        language: Literal["english", "german"] = "english",
        threads: int = 2,
        port: int = 23023,
        memory: int = 16,
    ):
        super().__init__(
            start_server=StartServer.FORCE_START,
            threads=threads,
            endpoint=f"http://localhost:{port}",
            annotators=[
                "tokenize",
                "pos",
                "lemma",
                "ner",
                "parse",
                "depparse",
            ],
            properties=language,
            output_format="conllu",
            memory=f"{memory}G",
            timeout=600000,
            be_quiet=True,
        )


class CoreNlpWorker(ParserABC):
    @staticmethod
    def process(
        task_queue: Queue,
        result_queue: PriorityQueue,
        port: int = 23023,
        **kwargs,
    ):
        properties = {
            "tokenize": "whitespace",
            "ssplit": "eolonly",
        } | kwargs

        with CoreNLPClient(
            start_server=StartServer.DONT_START,
            endpoint=f"http://127.0.0.1:{port}",
            timeout=600000,
            be_quiet=True,
        ) as client:
            while True:
                try:
                    match task_queue.get():
                        case (idx, text):
                            meta = ""
                        case (idx, text, meta):
                            pass
                        case invalid:
                            raise ValueError(f"Invalid item: {invalid}")
                    document: str = client.annotate(
                        text,
                        output_format="conllu",
                        properties=properties,
                    )
                    document = f"{meta}\n{document}".strip()
                    result_queue.put((idx, document))
                except Empty:
                    break
                except KeyboardInterrupt:
                    break
                except Exception as e:
                    raise ThreadError from e
                finally:
                    task_queue.task_done()
