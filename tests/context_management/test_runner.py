"""用于上下文管理测试的测试运行脚本。"""

import argparse
import subprocess
import sys
from pathlib import Path


def run_tests(test_type="all", verbose=False, coverage=False):
    """运行上下文管理测试。

    Args:
        test_type: 要运行的测试类型 ( 'all' , 'unit' , 'integration' , 'edge_cases') .
        verbose: 启用动词输出
        coverage: 启用覆盖报告
    """
    # 基础 pytest 命令
    cmd = ["python", "-m", "pytest"]

    # 测试目录
    test_dir = Path(__file__).parent

    # 根据类型添加特定测试文件
    if test_type == "all":
        cmd.append(str(test_dir))
    elif test_type == "unit":
        cmd.extend([str(test_dir / "test_context_manager.py"), str(test_dir / "test_context_config.py")])
    elif test_type == "integration":
        cmd.append(str(test_dir / "test_agent_graph_integration.py"))
    elif test_type == "edge_cases":
        cmd.extend([str(test_dir / "test_edge_cases.py"), str(test_dir / "test_state_operations.py")])
    else:
        print(f"Unknown test type: {test_type}")
        return False

    # 添加动词旗
    if verbose:
        cmd.append("-v")

    # 添加覆盖
    if coverage:
        cmd.extend(
            [
                "--cov=insightflow.context_manager",
                "--cov=insightflow.context_config",
                "--cov-report=html",
                "--cov-report=term-missing",
            ]
        )

    # 添加其他有用的旗帜
    cmd.extend(
        [
            "--tb=short",  # 更短的追溯格式
            "-x",  # 停止第一次失败
            "--strict-markers",  # 严格标记检查
        ]
    )

    print(f"Running command: {' '.join(cmd)}")
    print("-" * 50)

    # 运行测试
    try:
        result = subprocess.run(cmd, check=False)
        return result.returncode == 0
    except KeyboardInterrupt:
        print("\nTests interrupted by user")
        return False


def main():
    """试跑者的主要功能。"""
    parser = argparse.ArgumentParser(description="Run context management tests")

    parser.add_argument(
        "--type",
        "-t",
        choices=["all", "unit", "integration", "edge_cases"],
        default="all",
        help="Type of tests to run (default: all)",
    )

    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose output")

    parser.add_argument("--coverage", "-c", action="store_true", help="Enable coverage reporting")

    args = parser.parse_args()

    success = run_tests(test_type=args.type, verbose=args.verbose, coverage=args.coverage)

    if success:
        print("\n✅ All tests passed!")
        sys.exit(0)
    else:
        print("\n❌ Some tests failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()
