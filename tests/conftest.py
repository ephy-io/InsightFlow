"""Pytest配置和共享固定装置。"""

import os
import tempfile
from collections.abc import Generator
from pathlib import Path
from typing import Any
from unittest.mock import Mock

import pytest
from langchain_core.language_models import FakeListChatModel
from langchain_core.messages import AIMessage, HumanMessage
from sqlalchemy import create_engine

# 在测试收集过程中避免装入开发者的本地 InsightFlow 配置。
os.environ.setdefault("CONFIG_FILE", str(Path(tempfile.gettempdir()) / "insightflow-test-missing.yaml"))
os.environ.setdefault("INSIGHTFLOW_TEST_MODE", "true")

from insightflow.catalog.store.file_system import FileSystemCatalogStore
from insightflow.config_loader import ConfigLoader
from insightflow.graph_state import AgentState


@pytest.fixture(scope="session")
def test_config() -> dict[str, Any]:
    """测试配置 fixture ."""
    return {
        "organization": "TestOrg",
        "dialect": "presto",
        "bi_config_file": "test_bi.yaml",
        "catalog_store": {"store_type": "file_system", "data_path": "./test_data"},
        "default_llm": {
            "class": "langchain_core.language_models.FakeListChatModel",
            "params": {"responses": ["Test response"]},
        },
    }


@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """临时目录 fixture ."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield Path(temp_dir)


@pytest.fixture
def mock_llm() -> FakeListChatModel:
    """用于测试的mock LLM fixture ."""
    return FakeListChatModel(
        responses=["SELECT COUNT(*) FROM test_table;", "This is a test SQL query.", "Test analysis result."]
    )


@pytest.fixture
def sample_agent_state() -> AgentState:
    """样本代理状态 测试。"""
    return AgentState(
        messages=[HumanMessage(content="Test query")],
        sql="SELECT * FROM test_table;",
        agent_next_node="sql_generation",
        final_answer="Test data results",
    )


@pytest.fixture
def mock_catalog_store(temp_dir: Path) -> FileSystemCatalogStore:
    """Mock目录存储 fixture ."""
    # 创建测试数据文件
    test_data_dir = temp_dir / "test_data"
    test_data_dir.mkdir(exist_ok=True)

    # 创建样本 table_columns.csv
    tables_info_file = test_data_dir / "table_info.yaml"
    tables_info_file.write_text("""test:
  test_table:
    type: fact
    description: A test table for unit tests
  user_data:
    type: fact
    description: User information table""")

    # 创建样本 table_columns.csv
    tables_file = test_data_dir / "table_columns.csv"
    tables_file.write_text("""db_name,table_name,column_name
test,test_table,id
test,test_table,name
test,user_data,user_id""")

    # 创建样本 table_spec_columns.csv
    columns_file = test_data_dir / "table_spec_columns.csv"
    columns_file.write_text("""db_name,table_name,column_name,type,display_name,description
test,test_table,id,bigint,Id,Primary key
test,test_table,name,varchar,Name,User name
test,user_data,user_id,bigint,User Id,User identifier""")

    # 创建样本 common_columns.csv
    common_columns_file = test_data_dir / "common_columns.csv"
    common_columns_file.write_text("""column_name,type,display_name,description
status,varchar,Status,Record status
created_at,timestamp,Created At,Creation timestamp
updated_at,timestamp,Updated At,Last update timestamp""")

    # 模拟数据仓库配置
    data_warehouse_config = {"uri": "sqlite:///:memory:", "include_tables": None, "database_name": "test_db"}

    return FileSystemCatalogStore(data_path=str(test_data_dir), data_warehouse_config=data_warehouse_config)


@pytest.fixture
def mock_database_engine():
    """mock数据库引擎 fixture ."""
    engine = create_engine("sqlite:///:memory:")

    # 创建测试表格
    with engine.connect() as conn:
        conn.execute("CREATE TABLE test_table (id INTEGER, name TEXT)")
        conn.execute("INSERT INTO test_table VALUES (1, 'Test User')")
        conn.commit()

    return engine


@pytest.fixture
def sample_table_info() -> dict[str, Any]:
    """样本表信息 fixture ."""
    return {
        "test_table": {
            "columns": [
                {"name": "id", "type": "bigint", "description": "Primary key"},
                {"name": "name", "type": "varchar", "description": "User name"},
            ],
            "description": "A test table for unit tests",
            "sql_rule": "Always filter by active status",
        }
    }


@pytest.fixture
def sample_messages() -> list:
    """样本消息历史 fixture ."""
    return [
        HumanMessage(content="What's the user count?"),
        AIMessage(content="I'll help you get the user count from the database."),
        HumanMessage(content="Show me the SQL query"),
    ]


@pytest.fixture(autouse=True)
def reset_config_loader():
    """在每次测试前重置 ConfigLoader 单子状态。"""
    # 重置单顿实例以确保干净状态
    ConfigLoader._instance = None
    ConfigLoader._config = None
    yield
    # 测试后清理
    ConfigLoader._instance = None
    ConfigLoader._config = None


@pytest.fixture
def mock_config():
    """为需要的测试提供模拟配置。"""
    from unittest.mock import MagicMock

    config_dict = {
        "organization": "Test Company",
        "dialect": "presto",
        "default_llm": MagicMock(),
        "embedding_model": MagicMock(),
        "data_warehouse_config": {"uri": "sqlite:///:memory:", "include_tables": None, "database_name": "test_db"},
        "report_directory": "./data",
        "python_executor": "local",
        "visualization_mode": "rule",
        "context_config": {},
    }

    loader = ConfigLoader()
    loader.set(config_dict)
    return loader.get()


@pytest.fixture(autouse=True)
def setup_test_env(monkeypatch, temp_dir):
    """设置测试环境变量。"""
    monkeypatch.setenv("insightflow_CONFIG_PATH", str(temp_dir / "config.yaml"))
    monkeypatch.setenv("INSIGHTFLOW_TEST_MODE", "true")


class MockTokenService:
    """模拟信使服务测试。"""

    def __init__(self):
        self.token = "mock_token_12345"

    def get_token(self) -> str:
        return self.token


@pytest.fixture
def mock_token_service() -> MockTokenService:
    """(原始内容存档于2018-03-03). Moc sorry service fixture ."""
    return MockTokenService()


@pytest.fixture
def sample_sql_examples() -> list:
    """SQL 样本例 fixture ."""
    return [
        {"question": "How many users are there?", "sql": "SELECT COUNT(*) FROM users;", "tables": ["users"]},
        {
            "question": "What's the average age?",
            "sql": "SELECT AVG(age) FROM users WHERE age IS NOT NULL;",
            "tables": ["users"],
        },
    ]


@pytest.fixture
def mock_presto_connection():
    """Mock Presto 连接 fixture ."""
    mock_conn = Mock()
    mock_cursor = Mock()

    # 设置光标行为
    mock_cursor.fetchall.return_value = [("table1", "Test table 1"), ("table2", "Test table 2")]
    mock_cursor.description = [("table_name",), ("description",)]

    mock_conn.cursor.return_value = mock_cursor
    mock_conn.execute.return_value = mock_cursor

    return mock_conn
