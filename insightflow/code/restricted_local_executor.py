import sys
from io import StringIO
from typing import Any

from RestrictedPython import compile_restricted, safe_globals, utility_builtins  # type: ignore[attr-defined]
from RestrictedPython.Guards import safe_builtins, safer_getattr

from insightflow.code.executor_base import ExecutorBase


class RestrictedLocalExecutor(ExecutorBase):

    def run_code(self, code: str) -> tuple[bool, str]:
        try:
            # 编译限制代码
            byte_code = compile_restricted(code, "<string>", "exec")
            restricted_locals: dict[str, Any] = {}
            restricted_globals: dict[str, Any] = safe_globals.copy()

            # 设置具有必要功能的限制性环境
            restricted_globals.update(safe_builtins)
            restricted_globals["_getattr_"] = safer_getattr
            restricted_globals["__builtins__"] = utility_builtins

            # 在限制的本地定义中添加变量定义
            for key, value in self._variable.items():
                restricted_locals[key] = value

            # 抓取打印输出
            original_stdout = sys.stdout
            output_buffer = StringIO()
            sys.stdout = output_buffer

            # 使用 RestrictedPython 的标准打印函数
            restricted_globals["_print_"] = lambda *args, **kwargs: print(*args, **kwargs)

            # RestrictedPython 编译后有意执行边界。
            exec(byte_code, restricted_globals, restricted_locals)  # nosec B102
            output = output_buffer.getvalue()

            return True, output

        except Exception as e:
            return False, str(e)
        finally:
            if "original_stdout" in locals():
                sys.stdout = original_stdout
