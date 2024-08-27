from parse.supar_ import SuparParser
from parse.tests.core import IntegrationTestCohaABC, TestCohaABC, TestDtaABC


class SuparBiaffineTestCoha(TestCohaABC):
    def setUp(self):
        self.runner = SuparParser("biaffine", language="en")


class SuparBiaffineTestDta(TestDtaABC):
    def setUp(self):
        self.runner = SuparParser("biaffine", language="de")


class SuparBiaffineIntegrationTest(IntegrationTestCohaABC):
    def setUp(self):
        self.runner = SuparParser("biaffine", language="en", device="cuda:0")
