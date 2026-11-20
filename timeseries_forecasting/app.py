"""app.py: FastAPI 应用变形器时间序列预测。"""

import logging
import time
from typing import Any

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from model_handler import TransformerModelHandler, get_model_handler
from pydantic import BaseModel, Field
from starlette.requests import Request

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 创建 FastAPI app
app = FastAPI(
    title="Transformer Time Series Forecasting API",
    description="A REST API for time series forecasting using Transformer model",
    version="1.0.0",
)


# 请求模式
class ForecastRequest(BaseModel):
    """预报请求模型。"""

    input: list[float | int | dict[str, Any]] = Field(
        ...,
        description="Time series data as list of numbers or structured data",
        example=[100, 102, 98, 105, 107, 103, 99, 101],
    )
    forecast_window: int = Field(default=24, ge=1, le=200, description="Number of future points to predict")
    input_len: int | None = Field(default=None, description="Optional input length limit")
    frequency: str = Field(default="hourly", description="Frequency of the time series (hourly, daily, etc.)")
    target_column: str = Field(default="value", description="Column name for structured data")


class ForecastResponse(BaseModel):
    """预测的反应模型。"""

    predictions: list[float] = Field(description="Forecasted values")
    forecast_window: int = Field(description="Number of predictions")
    frequency: str = Field(description="Time series frequency")
    status: str = Field(description="Response status")


class ErrorResponse(BaseModel):
    """错误反应模型。"""

    error: str = Field(description="Error message")
    status: str = Field(description="Response status")


# 全球变量
model_handler: TransformerModelHandler | None = None
startup_time: float | None = None

# 配置属性可能带有模型最小输入长度，按优先级顺序排列。
# 对于Timer家族来说，这是 `input_token_len` (补丁长度);其他模型可能有所不同。
_MIN_INPUT_LENGTH_CONFIG_ATTRS = ("input_token_len", "context_length", "patch_len", "min_input_length")


def _resolve_min_input_length() -> int | None:
    """最好地读取模型的最小输入长度从其配置中需要的长度。

    当模型未初始化或不存在已知属性时返回 None ,所以调用者
    可以回到他们自己的默认状态。
    """
    if not model_handler or not model_handler.initialized or model_handler.config is None:
        return None
    for attr in _MIN_INPUT_LENGTH_CONFIG_ATTRS:
        value = getattr(model_handler.config, attr, None)
        if isinstance(value, int) and value > 0:
            return value
    return None


@app.on_event("startup")
async def startup_event():
    """启动时初始化模型。"""
    global model_handler, startup_time
    startup_time = time.time()
    logger.info("Starting Transformer Forecasting API...")

    try:
        # 初始化模型处理器
        model_handler = get_model_handler()
        model_success = model_handler.initialize()

        if model_success:
            logger.info("Model initialized successfully")
        else:
            logger.error("Failed to initialize model")

    except Exception as e:
        logger.error(f"Startup failed: {str(e)}")


@app.get("/health")
async def health_check():
    """健康检查终点。"""
    uptime = time.time() - startup_time if startup_time else 0

    return {
        "status": "healthy",
        "model_initialized": model_handler.initialized if model_handler else False,
        "min_input_length": _resolve_min_input_length(),
        "uptime_seconds": round(uptime, 2),
    }


@app.get("/ping")
async def ping():
    """简单的点。"""
    return {"status": "ok"}


@app.post(
    "/predict",
    response_model=ForecastResponse | ErrorResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Bad Request"},
        422: {"model": ErrorResponse, "description": "Validation Error"},
        500: {"model": ErrorResponse, "description": "Internal Error"},
    },
)
async def predict(request: ForecastRequest):
    """主预报终点。

    Args:
        request: 包含时间序列数据和参数的预测请求

    Returns:
        预测或错误的预测反应
    """
    try:
        logger.info(f"Received prediction request: {len(request.input)} data points, horizon={request.forecast_window}")

        # 检查模型是否初始化
        if not model_handler or not model_handler.initialized:
            raise HTTPException(status_code=500, detail="Model not initialized")

        # 校验输入
        if len(request.input) == 0:
            raise HTTPException(status_code=400, detail="Input data cannot be empty")

        # 做预测
        result = model_handler.predict(
            time_series_data=request.input,
            forecast_window=request.forecast_window,
            input_len=request.input_len,
            frequency=request.frequency,
            target_column=request.target_column,
        )

        # 检查预测是否成功
        if result.get("status") == "error":
            raise HTTPException(status_code=result.get("code", 500), detail=result.get("error", "Prediction failed"))

        logger.info(f"Prediction successful: {len(result['predictions'])} predictions generated")

        return ForecastResponse(**result)

    except HTTPException as e:
        return JSONResponse(status_code=e.status_code, content=ErrorResponse(error=str(e), status="error").model_dump())
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        return JSONResponse(status_code=500, content=ErrorResponse(error=str(e), status="error").model_dump())


@app.get("/model/info")
async def model_info():
    """获取模型信息。"""
    if not model_handler or not model_handler.initialized:
        return {"error": "Model not initialized", "status": "error"}

    return {
        "model_path": model_handler.model_path,
        "device": str(model_handler.device),
        "initialized": model_handler.initialized,
        "min_input_length": _resolve_min_input_length(),
        "config": model_handler.config.to_dict() if model_handler.config else None,
    }


@app.get("/")
async def root():
    """根端点带有 API 信息。"""
    return {
        "name": "Transformer Time Series Forecasting API",
        "version": "1.0.0",
        "description": "REST API for time series forecasting using Transformer model",
        "endpoints": {
            "predict": "/predict",
            "health": "/health",
            "ping": "/ping",
            "model_info": "/model/info",
            "docs": "/docs",
        },
    }


# 错误处理器
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """处理 HTTP 异常。"""
    return JSONResponse(
        status_code=exc.status_code,
        content={"status": "error", "message": exc.detail},
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """处理一般异常。"""
    logger.error(f"Unhandled exception: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content={"status": "error", "message": "Internal server error"},
    )


if __name__ == "__main__":
    # 促进发展
    uvicorn.run("app:app", host="0.0.0.0", port=8765, reload=True, log_level="info")
