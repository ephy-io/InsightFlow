FROM python:3.10-slim

# 只安装必要的构建工具
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    wget \
    build-essential \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# 安装时间序列预测所需的 Python 依赖
RUN pip3 --no-cache-dir install \
    fastapi==0.120.4 \
    uvicorn==0.38.0 \
    transformers==4.40.1 \
    torch==2.9.0 \
    numpy==2.2.6 \
    pandas==2.3.3 \
    pydantic==2.12.3

# 设置工作目录
WORKDIR /home/model-server

# 复制模型文件
COPY ../hf_model /home/model-server/hf_model

# 复制应用文件
COPY app.py model_handler.py /home/model-server/

# 设置环境变量
ENV PYTHONPATH=/home/model-server
ENV PYTHONUNBUFFERED=1

# 声明服务端口
EXPOSE 8765

# 设置容器入口和默认命令
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8765"]