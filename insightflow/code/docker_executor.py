import logging
import os
import shutil

# 仅用于固定的 Docker CLI 健康检查。
import subprocess  # nosec B404
import tempfile
from pathlib import Path
from typing import Any

import docker
from docker.errors import ContainerError

from insightflow.code.executor_base import ExecutorBase

logger = logging.getLogger(__name__)


def _get_docker_cli_path() -> str | None:
    """解析 Docker CLI 一旦如此子进程，则不依赖部分可执行路径。"""
    return shutil.which("docker")


def check_docker_status() -> tuple[bool, str]:
    """检查 Docker 的安装和状态，而不初始化 DockerExecutor .

    Returns:
        Tuple[bool, str]: ( is_available , status_message) (中文(简体)).
    """
    try:
        # 检查是否安装了 Docker CLI
        docker_cli_path = _get_docker_cli_path()
        if not docker_cli_path:
            return False, "Docker is not installed. Please install Docker."

        # 检查 Docker 守护进程是否运行
        # 固定命令和参数;没有用户输入到达子进程。
        result = subprocess.run([docker_cli_path, "info"], capture_output=True, text=True, timeout=10)  # nosec B603

        if result.returncode == 0:
            return True, "Docker is installed and running"
        else:
            if "Cannot connect to the Docker daemon" in result.stderr:
                return False, "Docker is installed but not running. Please start the Docker daemon."
            else:
                return False, f"Docker is not available: {result.stderr.strip()}"

    except subprocess.TimeoutExpired:
        return False, "Docker command timed out. Docker may not be running properly."
    except FileNotFoundError:
        return False, "Docker command not found. Please install Docker."
    except Exception as e:
        return False, f"Error checking Docker status: {str(e)}"


class DockerExecutor(ExecutorBase):
    """Docker -基于 Python 代码执行器用于孤立执行。"""

    def __init__(self, variable: dict[str, Any] | None = None) -> None:
        super().__init__(variable)
        self.image_name = "python-executor"
        self.dockerfile_path = Path(__file__).parent.parent.parent / "Dockerfile.python-executor"

        # 检查 Docker 安装和状态
        self._check_docker_availability()

        try:
            self.client: Any = docker.from_env()
            # 如果不存在，则构建 Docker 图像
            self._ensure_image_exists()
        except Exception as e:
            self._handle_docker_error(e)

    @staticmethod
    def _check_docker_availability() -> None:
        """检查 Docker 是否安装和可用。"""
        # 检查是否安装了 Docker CLI
        docker_cli_path = _get_docker_cli_path()
        if not docker_cli_path:
            raise RuntimeError("Docker is not installed. Please install Docker and ensure it's in your system PATH.")

        # 检查 Docker 守护进程是否运行
        try:
            # 固定命令和参数;没有用户输入到达子进程。
            result = subprocess.run([docker_cli_path, "info"], capture_output=True, text=True, timeout=10)  # nosec B603
            if result.returncode != 0:
                if "Cannot connect to the Docker daemon" in result.stderr:
                    raise RuntimeError(
                        "Docker is installed but not running. Please start the Docker daemon and try again."
                    )
                else:
                    raise RuntimeError(
                        f"Docker is not available. Please check Docker installation and status. "
                        f"Error: {result.stderr.strip()}"
                    )
        except subprocess.TimeoutExpired:
            raise RuntimeError("Docker command timed out. Please check if Docker is running properly.") from None
        except FileNotFoundError:
            raise RuntimeError(
                "Docker command not found. Please install Docker and ensure it's in your system PATH."
            ) from None

    @staticmethod
    def _handle_docker_error(error: Exception) -> None:
        """处理 Docker - 带有特定错误消息的相关错误。"""
        error_str = str(error).lower()

        if "connection aborted" in error_str and "no such file or directory" in error_str:
            raise RuntimeError("Docker is not running. Please start the Docker daemon and try again.")
        elif "permission denied" in error_str:
            raise RuntimeError(
                "Permission denied accessing Docker. Please ensure your user has Docker permissions "
                "or try running with appropriate privileges."
            )
        elif "docker daemon" in error_str or "connection refused" in error_str:
            raise RuntimeError("Cannot connect to Docker daemon. Please start the Docker daemon and try again.")
        else:
            raise RuntimeError(
                f"Failed to initialize Docker client. Please ensure Docker is installed and running. "
                f"Error: {str(error)}"
            )

    def _ensure_image_exists(self) -> None:
        """如果不存在，则构建 Docker 图像。"""
        try:
            self.client.images.get(self.image_name)
        except docker.errors.ImageNotFound:
            print(f"Building Docker image '{self.image_name}'...")
            self.client.images.build(
                path=str(self.dockerfile_path.parent),
                dockerfile=self.dockerfile_path.name,
                tag=self.image_name,
                rm=True,
            )
            print(f"Docker image '{self.image_name}' built successfully.")

    def run_code(self, code: str) -> tuple[bool, str]:
        """在 Docker 容器中执行 Python 代码。"""
        try:
            # 创建一个带有代码的临时文件
            with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
                # 在代码中添加可变定义
                variable_code = ""
                for key, value in self._variable.items():
                    if isinstance(value, str):
                        variable_code += f'{key} = "{value}"\n'
                    else:
                        variable_code += f"{key} = {repr(value)}\n"

                full_code = variable_code + "\n" + code
                f.write(full_code)
                temp_file_path = f.name

            try:
                # 在 Docker 容器中运行代码
                container = self.client.containers.run(
                    self.image_name,
                    command=["python3", f"/app/{os.path.basename(temp_file_path)}"],
                    volumes={temp_file_path: {"bind": f"/app/{os.path.basename(temp_file_path)}", "mode": "ro"}},
                    remove=True,
                    detach=False,
                    stdout=True,
                    stderr=True,
                    network_mode="none",  # 禁用网络安全访问
                )

                # 获取输出
                output = container.decode("utf-8")
                return True, output

            except ContainerError as e:
                # 集装箱以非零码退出
                error_output = e.stderr if e.stderr else str(e)
                return False, f"Container execution failed: {error_output}"

        except Exception as e:
            return False, f"Docker execution error: {str(e)}"

        finally:
            # 清理临时文件
            if "temp_file_path" in locals() and os.path.exists(temp_file_path):
                try:
                    os.unlink(temp_file_path)
                except (OSError, PermissionError) as e:
                    # 日志，但不要因为清理问题而失败操作
                    print(f"Warning: Failed to clean up temporary file {temp_file_path}: {e}")

    def __del__(self) -> None:
        """在删除时清理 Docker 客户端。"""
        try:
            if hasattr(self, "client") and self.client is not None:
                self.client.close()
        except Exception as e:
            # 在对象销毁时忽略清理错误
            logger.debug("Failed to close Docker client during cleanup: %s", e)
