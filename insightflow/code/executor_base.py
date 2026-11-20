from typing import Any


class ExecutorBase:
    """用于执行蟒蛇代码的基班。"""

    _variable: dict[str, Any]

    def __init__(self, variable: dict[str, Any] | None = None) -> None:
        if variable is None:
            self._variable = {}
        else:
            self._variable = variable

    def run_code(self, code: str) -> tuple[bool, str]:
        """执行蟒蛇代码。"""
        raise NotImplementedError()

    def set_variable(self, key: str, value: Any) -> None:
        """设定变量。"""
        self._variable[key] = value
