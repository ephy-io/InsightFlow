#!/usr/bin/env python3
"""Streamlit -基于 InsightFlow 接口的启动脚本。

用法 :
    python run_streamlit_ui.py (中文(简体)).

这将启动 Streamlit 服务器于 http://localhost:8501
"""

import os
import subprocess
import sys


def main():
    """发射 Streamlit UI"""
    # 更改项目目录
    project_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(project_dir)

    print("🚀 Starting InsightFlow Streamlit UI...")
    print("📍 URL: http://localhost:8501")
    print("⏹️  Press Ctrl+C to stop the server")
    print("-" * 50)

    try:
        # 用新 UI 文件运行流格
        subprocess.run(
            [
                sys.executable,
                "-m",
                "streamlit",
                "run",
                "sample_ui/streamlit_ui.py",
                "--server.port=8501",
                "--server.address=localhost",
            ],
            check=True,
        )
    except KeyboardInterrupt:
        print("\n👋 Stopping Streamlit server...")
    except subprocess.CalledProcessError as e:
        print(f"❌ Error starting Streamlit: {e}")
        print("\n💡 Make sure Streamlit is installed:")
        print("   pip install streamlit")
    except FileNotFoundError:
        print("❌ Python or Streamlit not found")
        print("\n💡 Make sure Python and Streamlit are installed:")
        print("   pip install streamlit")


if __name__ == "__main__":
    main()
