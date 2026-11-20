"""整个 InsightFlow 应用程序使用的常数。"""

# 日期/时间格式字符串
datetime_format = "%Y-%m-%d %H:%M:%S"
date_format = "%Y-%m-%d"
datetime_format_ms = "%Y-%m-%d %H:%M:%S.%f"
datetime_format_ms_T = "%Y-%m-%dT%H:%M:%S.%fZ"

# SQL 执行状态代码
SQL_NA = "SQL_NA"
SQL_SUCCESS = "SQL_SUCCESS"
SQL_EXECUTE_TIMEOUT = "SQL_CHECK_TIMEOUT"
SQL_SYNTAX_ERROR = "SQL_SYNTAX_ERROR"
SQL_SECURITY_ERROR = "SQL_SECURITY_ERROR"
SQL_UNKNOWN_ERROR = "SQL_UNKNOWN_ERROR"
SQL_RESULT_LIMIT = 10000

# 默认工具回合( search_knowledge 或 AskHuman) 信息提取可能用于一个问题
EXTRACTION_MAX_TOOL_ROUNDS = 10

MCP_TOOL_DEFAULT_TIMEOUT_SECONDS = 60
