"""model_handler.py:用于时间序列预测的基于变形器的模型处理器。"""

import logging
from typing import Any

import numpy as np
import pandas as pd
import torch
from transformers import AutoConfig, AutoModelForCausalLM

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TransformerModelHandler:
    """基于变形器的模型处理器用于时间序列预测。"""

    def __init__(self, model_path: str = "hf_model"):
        """初始化模型处理器。"""
        logger.info("Initializing Transformer Model Handler")
        self.model_path = model_path
        self.model = None
        self.config = None
        self.device = None
        self.initialized = False

    def initialize(self) -> bool:
        """初始化型号。

        Returns:
            bool: True 如果初始化成功
        """
        try:
            logger.info("Starting model initialization")

            # 设置设备
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            logger.info(f"Using device: {self.device}")

            logger.info(f"Loading model from: {self.model_path}")

            # 装入模型配置
            self.config = AutoConfig.from_pretrained(self.model_path, trust_remote_code=True)

            # 装入预先训练过的模型
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path, config=self.config, trust_remote_code=True
            )

            # 将模型移动到设备
            self.model.to(self.device)
            self.model.eval()

            self.initialized = True
            logger.info("Transformer model loaded successfully")
            logger.info(f"Model config: {self.config}")

            return True

        except Exception as e:
            logger.error(f"Failed to initialize model: {str(e)}")
            self.initialized = False
            return False

    def preprocess(
        self,
        time_series_data: list,
        forecast_window: int = 24,
        input_len: int | None = None,
        frequency: str = "hourly",
        target_column: str = "value",
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """将原始输入转换为模型输入数据。

        Args:
            time_series_data: 输入时间序列数据
            forecast_window: 未来预测的点数
            input_len: 可选输入长度限制
            frequency: 时间序列的频率
            target_column: 结构化数据的列名

        Returns:
            Tuple of ( processed_tensor ,元数据) (中文(简体)).
        """
        try:
            logger.info(f"Input data length: {len(time_series_data) if isinstance(time_series_data, list) else 'N/A'}")
            logger.info(f"Forecast window: {forecast_window}")

            # 将输入转换为数字数组
            if isinstance(time_series_data, list):
                if len(time_series_data) > 0 and isinstance(time_series_data[0], dict):
                    # 处理结构化数据(带有时间戳)
                    df = pd.DataFrame(time_series_data)
                    if target_column in df.columns:
                        values = df[target_column].values
                    else:
                        # 使用第一个数字列
                        numeric_cols = df.select_dtypes(include=[np.number]).columns
                        if len(numeric_cols) > 0:
                            values = df[numeric_cols[0]].values
                        else:
                            values = np.array([float(x) for x in time_series_data])
                else:
                    # 处理简单的数字列表
                    values = np.array([float(x) for x in time_series_data])
            else:
                values = np.array(time_series_data)

            # 处理输入长度限制
            if input_len is not None:
                if input_len > len(values):
                    # 如果输入比需要短，则以零为标签
                    values = np.pad(values, (input_len - len(values), 0), mode="constant", constant_values=0)
                elif input_len < len(values):
                    # 取最后一个 input_len 值
                    values = values[-input_len:]

            # 使数据正常化(简单Z分数正常化)
            mean_val = np.mean(values)
            std_val = np.std(values)
            if std_val > 0:
                normalized_values = (values - mean_val) / std_val
            else:
                normalized_values = values - mean_val

            # 转换为亮度
            tensor = torch.tensor(normalized_values, dtype=torch.float32).unsqueeze(0)
            tensor = tensor.to(self.device)

            # 为处理后存储元数据
            metadata = {
                "mean": mean_val,
                "std": std_val,
                "forecast_window": forecast_window,
                "frequency": frequency,
                "original_length": len(values),
            }

            logger.info(f"Preprocessed tensor shape: {tensor.shape}")

            return tensor, metadata

        except Exception as e:
            logger.error(f"Preprocessing failed: {str(e)}")
            raise e

    def inference(self, input_tensor: torch.Tensor, metadata: dict[str, Any]) -> torch.Tensor:
        """运行推论模型。

        Args:
            input_tensor: 预处理的输入分数
            metadata: 预处理元数据

        Returns:
            模型输出亮度
        """
        try:
            if not self.initialized:
                raise RuntimeError("Model not initialized")

            with torch.no_grad():
                forecast_window = metadata.get("forecast_window", 24)

                # 使用生成方法
                output = self.model.generate(input_tensor, max_new_tokens=forecast_window)

                logger.info(f"Model output shape: {output.shape}")
                return output

        except ValueError as e:
            logger.error(f"Inference failed due to ValueError: {str(e)}")
            raise e
        except Exception as e:
            logger.error(f"Inference failed: {str(e)}")
            raise e

    def postprocess(self, output_tensor: torch.Tensor, metadata: dict[str, Any]) -> list[float]:
        """将模型输出转换为最终预测格式。

        Args:
            output_tensor: 原始模型输出
            metadata: 预处理元数据

        Returns:
            作为清单的最后预测
        """
        try:
            # 从百分数中提取预测
            if output_tensor.dim() > 1:
                predictions = output_tensor[0].cpu().numpy()
            else:
                predictions = output_tensor.cpu().numpy()

            # 使预测非正常化
            mean_val = metadata.get("mean", 0)
            std_val = metadata.get("std", 1)

            if std_val > 0:
                denormalized_predictions = predictions * std_val + mean_val
            else:
                denormalized_predictions = predictions + mean_val

            # 转换为列表，并确保其长度是合适的
            forecast_window = metadata.get("forecast_window", 24)
            result = denormalized_predictions[:forecast_window].tolist()

            logger.info(f"Final predictions length: {len(result)}")

            return result

        except Exception as e:
            logger.error(f"Postprocessing failed: {str(e)}")
            raise e

    def predict(
        self,
        time_series_data: list,
        forecast_window: int = 24,
        input_len: int | None = None,
        frequency: str = "hourly",
        target_column: str = "value",
    ) -> dict[str, Any]:
        """主要预测方法。

        Args:
            time_series_data: 输入时间序列数据
            forecast_window: 未来预测的点数
            input_len: 可选输入长度限制
            frequency: 时间序列的频率
            target_column: 结构化数据的列名

        Returns:
            包含预测和元数据的词典
        """
        try:
            # 确保模式初始化
            if not self.initialized:
                if not self.initialize():
                    raise RuntimeError("Failed to initialize model")

            # 预处理输入
            input_tensor, metadata = self.preprocess(
                time_series_data, forecast_window, input_len, frequency, target_column
            )

            # 运行推断
            output_tensor = self.inference(input_tensor, metadata)

            # 进程后输出
            predictions = self.postprocess(output_tensor, metadata)

            # 格式结果
            result = {
                "predictions": predictions,
                "forecast_window": metadata.get("forecast_window", 24),
                "frequency": metadata.get("frequency", "hourly"),
                "status": "success",
            }

            return result

        except ValueError as e:
            logger.error(f"Prediction failed due to ValueError: {str(e)}")
            return {"error": str(e), "code": 400, "status": "error"}
        except Exception as e:
            logger.error(f"Prediction failed: {str(e)}")
            return {"error": str(e), "status": "error"}


# 全球模型处理器实例
_model_handler = None


def get_model_handler() -> TransformerModelHandler:
    """获取或创建全局模型处理器实例。"""
    global _model_handler
    if _model_handler is None:
        _model_handler = TransformerModelHandler()
    return _model_handler
