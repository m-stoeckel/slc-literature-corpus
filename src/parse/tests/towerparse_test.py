from parse.towerparse_ import TowerParseRunner
from parse.tests.core import IntegrationTestCohaABC, TestCohaABC, TestDtaABC


class TowerParseTestCoha(TestCohaABC):
    def setUp(self):
        self.runner = TowerParseRunner(language="en")


class TowerParseTestDta(TestDtaABC):
    def setUp(self):
        self.runner = TowerParseRunner(language="de")


class TowerParseIntegrationTest(IntegrationTestCohaABC):
    def setUp(self):
        self.runner = TowerParseRunner(language="en", device="cuda:0")
