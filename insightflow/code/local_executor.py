import sys
from io import StringIO

from insightflow.code.executor_base import ExecutorBase


class LocalExecutor(ExecutorBase):

    def run_code(self, code: str) -> tuple[bool, str]:
        safe_globals = {"__builtins__": __builtins__}
        original_stdout = sys.stdout
        output_buffer = StringIO()
        sys.stdout = output_buffer
        try:
            # 有意本地代码执行器:呼叫者只能通过信任的代码。
            exec(code, safe_globals, safe_globals)  # nosec B102
            output = output_buffer.getvalue()
            return True, output
        except Exception as e:
            return False, str(e)
        finally:
            sys.stdout = original_stdout
