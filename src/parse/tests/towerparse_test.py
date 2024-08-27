from parse.towerparse_ import TowerParser
from parse.tests.core import IntegrationTestCohaABC, TestCohaABC, TestDtaABC


class TowerParseTestCoha(TestCohaABC):
    def setUp(self):
        self.runner = TowerParser(language="en", max_sentence_len=30)


class TowerParseTestDta(TestDtaABC):
    def setUp(self):
        self.runner = TowerParser(language="de")


class TowerParseIntegrationTest(IntegrationTestCohaABC):
    def setUp(self):
        self.runner = TowerParser(language="en", device="cuda:0")
