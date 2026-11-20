#!/bin/bash

# 构建并启动时间序列预测服务
set -e

echo "=== Building Timeseries Forecasting Docker Container ==="

# 检查 hf_model 模型目录是否存在
MODEL_DIR="../hf_model"
if [ ! -d "$MODEL_DIR" ]; then
    echo "Error: Hugging face model directory not found at $MODEL_DIR"
    echo "Please ensure the model is downloaded and available at this location"
    exit 1
fi

echo "✓ Found Hugging face model at: $MODEL_DIR"

rm -rf hf_model
cp -r $MODEL_DIR .

# 构建 Docker 镜像
echo "Building Docker image..."
docker build -t timeseries-forecasting .

if [ $? -eq 0 ]; then
    echo "✓ Docker image built successfully"
else
    echo "✗ Failed to build Docker image"
    exit 1
fi

# 检查是否存在同名容器，包含运行和停止状态，找到后移除
CONTAINER_NAME="time-series-forecasting-service"
if [ "$(docker ps -aq -f name=^/${CONTAINER_NAME}$)" ]; then
    echo "Removing existing container..."
    docker rm -f $CONTAINER_NAME
fi

echo "=== Starting Timeseries Forecasting Service ==="

# 启动容器
docker run -d \
    --name $CONTAINER_NAME \
    -p 8765:8765 \
    timeseries-forecasting

if [ $? -eq 0 ]; then
    echo "✓ Container started successfully"
    echo ""
    echo "Service endpoints:"
    echo "  - Predictions: http://localhost:8765/predict"
    echo "  - Health Check: http://localhost:8765/health"
    echo "  - API Docs: http://localhost:8765/docs"
    echo ""
    echo "Container logs:"
    echo "  docker logs -f $CONTAINER_NAME"
    echo ""
    echo "To test the service:"
    echo "  python test_forecasting.py"
    echo ""
    echo "To stop the service:"
    echo "  docker stop $CONTAINER_NAME"
else
    echo "✗ Failed to start container"
    exit 1
fi

# 稍等片刻，确认容器仍在运行
sleep 5
if [ "$(docker ps -q -f name=$CONTAINER_NAME)" ]; then
    echo "✓ Service is running"

    # 输出近期日志
    echo ""
    echo "=== Initial Service Logs ==="
    docker logs "$CONTAINER_NAME" | head -n 50
else
    echo "✗ Service failed to start"
    echo "Checking logs..."
    docker logs $CONTAINER_NAME
    exit 1
fi