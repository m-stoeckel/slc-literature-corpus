from parse.stackpointer_ import StackPointerRunner
from parse.tests.core import IntegrationTestCohaABC, TestCohaABC, TestDtaABC


class StackPointerTestCoha(TestCohaABC):
    def setUp(self):
        self.runner = StackPointerRunner(language="en")


class StackPointerTestDta(TestDtaABC):
    def setUp(self):
        self.runner = StackPointerRunner(language="de")


class StackPointerIntegrationTest(IntegrationTestCohaABC):
    def setUp(self):
        self.runner = StackPointerRunner(language="en", device="cuda:0")
