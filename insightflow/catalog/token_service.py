"""与外部服务进行认证的托肯服务。"""

import json

import requests

REQUEST_TIMEOUT_SECONDS = 10


class TokenService:
    """管理认证符的服务。

    处理令牌应用程序、验证和认证
    与外部服务。
    """

    base_url: str | None = None
    token: str | None = None
    user_name: str | None = None
    password: str | None = None

    def __init__(self, user_name: str, password: str):
        """初始化代号服务。"""
        self.user_name = user_name
        self.password = password

    def apply_token(self):
        """使用证书申请认证符。"""
        assert self.base_url is not None, "base_url must be set before calling apply_token"
        response = requests.post(
            self.base_url + "/apply_token",
            data=json.dumps({"user_name": self.user_name, "password": self.password}),
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
        resp_json = response.json()
        self.token = resp_json.get("token")


def apply_token_for_user(token_url: str, user_name: str, password: str):
    """使用用户名申请令牌并返回令牌。

    Args:
        token_url (str): 基础 URL 用于代币服务。
        user_name (str): 用户名。
        password (str): 密码

    Returns:
        符号
    """
    token_service = TokenService(user_name, password)
    token_service.base_url = token_url
    token_service.apply_token()
    return token_service.token
