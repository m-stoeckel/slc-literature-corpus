from parse.stackpointer_ import StackPointerParser
from parse.tests.core import IntegrationTestCohaABC, TestCohaABC, TestDtaABC


class StackPointerTestCoha(TestCohaABC):
    def setUp(self):
        self.runner = StackPointerParser(language="en", max_sentence_len=30)


class StackPointerTestDta(TestDtaABC):
    def setUp(self):
        self.runner = StackPointerParser(language="de")


class StackPointerIntegrationTest(IntegrationTestCohaABC):
    def setUp(self):
        self.runner = StackPointerParser(language="en", device="cuda:0")
