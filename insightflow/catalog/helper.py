from typing import Any

import requests
from sqlalchemy import Engine, create_engine

from insightflow.catalog.token_service import apply_token_for_user
from insightflow.utils import log


def get_requests_session(token: str, header_extra_params: dict) -> requests.Session:
    """以无记名符号认证创建 HTTP 会话。"""
    session = requests.Session()
    session.headers.update({"Authorization": f"Bearer {token}"})
    if header_extra_params:
        session.headers.update(header_extra_params)
    return session


def create_sqlalchemy_engine_instance(data_warehouse_config: dict[str, Any]) -> Engine:
    """从数据仓库配置创建 SQLAlchemy 引擎实例

    Args:
        data_warehouse_config: 配置 'uri' 和可选 'token_service'

    Returns:
        配置了 SQLAlchemy 引擎
    """
    database_uri: str = data_warehouse_config.get("uri") or ""

    engine_args: dict[str, Any] = {"echo": True}

    # 处理 Presto 认证
    if "presto" in database_uri and "token_service" in data_warehouse_config:
        token_service = str(data_warehouse_config.get("token_service") or "")
        user_name = str(data_warehouse_config.get("user_name") or "")
        password = str(data_warehouse_config.get("password") or "")
        header_extra_params = data_warehouse_config.get("header_extra_params", {})
        token = apply_token_for_user(token_service, user_name, password)
        log(f"Applied presto token: {token} for user: {user_name}")
        engine_args["connect_args"] = {
            "protocol": "https",
            "requests_session": get_requests_session(token, header_extra_params),
        }
        database_uri = database_uri.format(user_name=user_name)

    engine = create_engine(database_uri, **engine_args)

    return engine
