from parse.stanza_ import StanzaRunner
from parse.tests.core import IntegrationTestCohaABC, TestCohaABC, TestDtaABC


class StanzaTestCoha(TestCohaABC):
    def setUp(self):
        self.runner = StanzaRunner(language="en")


class StanzaTestDta(TestDtaABC):
    def setUp(self):
        self.runner = StanzaRunner(language="de")


class StanzaIntegrationTest(IntegrationTestCohaABC):
    def setUp(self):
        self.runner = StanzaRunner(language="en", device="cuda:0")
