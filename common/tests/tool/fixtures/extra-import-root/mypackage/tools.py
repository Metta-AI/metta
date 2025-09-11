from metta.common.tool import Tool


class TestTool(Tool):
    def invoke(self, args, overrides):
        print("TestTool invoked")
        return 0
