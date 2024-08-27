from parse.stanza_ import StanzaParser
from parse.tests.core import IntegrationTestCohaABC, TestCohaABC, TestDtaABC


class StanzaTestCoha(TestCohaABC):
    def setUp(self):
        self.runner = StanzaParser(language="en", max_sentence_len=30)


class StanzaTestDta(TestDtaABC):
    def setUp(self):
        self.runner = StanzaParser(language="de")


class StanzaIntegrationTest(IntegrationTestCohaABC):
    def setUp(self):
        self.runner = StanzaParser(language="en", device="cuda:0")


if __name__ == "__main__":
    StanzaIntegrationTest().run()