"""时间序列异常探测的核心算法。

评分战略遵循基于规则/战略的设计，其中每个因素都是：
(a) 故意正交，以便不重复计算同一证据：

- 偏离意义:实际值在统计学上距离
  以强烈噪声尺度(sigma)单位表示的预测。
  “ 如何异常” 信号并替换先前的冗余
  重建-error + 约束-违反对。
- 方向:下降对上升的加权可能不同，因为，对大多数
  商业衡量标准，意外下降比上升更严重。
- 音量调制:高流量时刻的异常比高流量时刻更重要
  低流量时刻的相对偏差相同。这是乘数驱动
  以预期(预估)水平表示，NOT 以异常实际值表示，所以
  滴至零绝不被它惩罚。
- 历史异常频率:违反自身标准的噪音/波动度量度
  界限经常受到抑制，从而减少虚假的阳性。
- 持续时间:窗口末端附近的连续异常点运行
  提振分数。

最后得分在[0, 1]之间; 接近1的数值表示更多
影响较大的异常情况。
"""

import math
from typing import Any

from insightflow import config
from insightflow.tool.timeseries_forecast import call_timeseries_service, check_forecast_service_health

# 偏差被认为达到 "bound" 的 sigma 数。
SIGMA_BOUND = 3.0
# 以每点严重性为临界值，高于此点则算作异常点。
ANOMALY_THRESHOLD = 0.6
# 默认方向权重( 中性)。 增加 DROP_WEIGHT 以强调下降。
DEFAULT_DROP_WEIGHT = 1.0
DEFAULT_RISE_WEIGHT = 1.0
# 下窗平滑强度在 [0, 1]。 窗口分数归结于
# 最后一点( 锁定);此旋钮将主锚本身的贡献与
# 0 = 纯锚(最敏感),
# 1 表示完全采用平滑均值，去噪程度最高。默认使用较低权重，使基准值
# 主导，同时加固孤立的单点尖顶。
DEFAULT_SMOOTHING = 0.5

# 上层( detection_range)的间隔默认值。
# 这些都是有意保守的起点，预计将是：
# 与真实的正常/异常样本相对应(见设计.md"开放问题").
# Hysteresis阈值:当严重性超过 T_HIGH 时打开一个间隔，
# 只在严重性低于 T_LOW ( T_LOW < T_HIGH)时关闭。
DEFAULT_T_HIGH = ANOMALY_THRESHOLD
DEFAULT_T_LOW = 0.4
# 由 MERGE_GAP 点分隔的相邻间隔被合并为一个。
DEFAULT_MERGE_GAP = 1
# 如果它的面积( 严重程度之和) QQ AREA_MIN OR 其峰值，则会保持间隔。
# right QQ PEAK_HIGH (后者是尖锐短短的高峰覆盖)
# 反常现象，否则会用一个仅长的过滤器来降低).
DEFAULT_AREA_MIN = 1.0
DEFAULT_PEAK_HIGH = 0.8

# 候选键用于从结构排列的行中浮出一个人可读的时间戳。
_TIMESTAMP_KEYS = ("timestamp", "time", "datetime", "date", "ts", "dt", "period")


def _default_evaluation_window(frequency: str) -> int:
    """返回给定频率的内部下窗大小。

    细颗粒是噪声，所以它们使用一个稍长的窗口
    denoise; 较紧的颗粒性保持短的默认值。
    """
    freq = (frequency or "").lower()
    if "min" in freq:
        return 6
    return 3


def _default_stride(frequency: str) -> int:
    """返回一个频率的默认块预示速度( = 地平线)。

    步法将每个区块的多步预测视野绑定:小到
    避免远视平滑，大到可以保持低调。
    周度或月度序列的样本较少，因此使用较小的步长。
    """
    freq = (frequency or "").lower()
    if "min" in freq:
        return 12
    if "hour" in freq:
        return 12
    if "day" in freq or freq == "daily":
        return 7
    if "week" in freq:
        return 4
    if "month" in freq:
        return 4
    return 10


def _extract_timestamp(item: Any) -> Any | None:
    """如果存在，从结构化行返回一个像时间戳的值。"""
    if isinstance(item, dict):
        for key in _TIMESTAMP_KEYS:
            if key in item:
                return item[key]
    return None


def _extract_values(data: list[float | int | dict[str, Any]], target_column: str = "value") -> list[float]:
    """从输入数据中提取数字值，默默跳过无效的条目。"""
    values: list[float] = []
    for item in data:
        if isinstance(item, bool):
            # bol是 int 的子类; 忽略以避免将 True / False 当作数据处理
            continue
        if isinstance(item, int | float):
            values.append(float(item))
        elif isinstance(item, dict) and target_column in item:
            try:
                values.append(float(item[target_column]))
            except (ValueError, TypeError):
                continue
    return values


def _mean(values: list[float]) -> float:
    """算术平均值; 返回 0.0为空输入。"""
    return sum(values) / len(values) if values else 0.0


def _median(values: list[float]) -> float:
    """中间值; 返回 0.0为空输入。"""
    if not values:
        return 0.0
    ordered = sorted(values)
    n = len(ordered)
    mid = n // 2
    if n % 2:
        return ordered[mid]
    return (ordered[mid - 1] + ordered[mid]) / 2


def _estimate_noise_scale(historical_values: list[float]) -> float:
    """估计一个系列的强烈噪声规模(sigma).

    研究第一差异(它消除趋势/季节性，所以尺度是：
    对于周期性度量表，不象原始水平标准那样膨胀)和
    使用基于 MAD 的估算器，以便不发生历史标记
    缩放比例表。标准偏差只在以下情况下作为后退使用：
    MAD 已退化(例如，多为恒定系列，单跳).
    """
    if len(historical_values) < 2:
        return 0.0

    diffs = [historical_values[i] - historical_values[i - 1] for i in range(1, len(historical_values))]
    if len(diffs) < 2:
        return abs(diffs[0]) / math.sqrt(2)

    median_diff = _median(diffs)
    mad = _median([abs(d - median_diff) for d in diffs])
    # 1.4826 缩放 MAD 与std 一致，用于正常数据。
    sigma_diff = 1.4826 * mad
    if sigma_diff <= 0:
        mean_diff = _mean(diffs)
        variance = sum((d - mean_diff) ** 2 for d in diffs) / (len(diffs) - 1)
        sigma_diff = math.sqrt(variance)
    return sigma_diff / math.sqrt(2)


def _deviation_significance(actual: float, predicted: float, sigma: float) -> tuple[float, float]:
    """返回 (显著性分数 [0, 1], 带符号的 z-score)。

    其意义在 SIGMA_BOUND (3 sigma) 绑定时线性增长到 0.5,
    然后线性到1.0,时速是束缚的两倍(6西格玛),在它饱和的地方。
    """
    # 楼层西格玛到一层的一小部分所以是一个近乎恒定的系列
    # 仍然对清除跳跃反应，而不以~0除以。
    effective_sigma = max(sigma, abs(predicted) * 1e-3, 1e-9)
    z = (actual - predicted) / effective_sigma
    abs_z = abs(z)

    if abs_z <= SIGMA_BOUND:
        significance = 0.5 * (abs_z / SIGMA_BOUND)
    else:
        significance = 0.5 + 0.5 * min((abs_z - SIGMA_BOUND) / SIGMA_BOUND, 1.0)

    return significance, z


def _volume_factor(expected_level: float, historical_values: list[float]) -> float:
    """[0.6,1.0]中根据预期流量水平的调值乘数。

    高流量瞬间保持全重(1.0);低流量瞬间被挡住
    受预测/预期水平的驱动，导致异常情况
    实际值(如降至零)从不减少因数。
    """
    hist_mean = _mean(historical_values)
    if hist_mean <= 0:
        return 1.0
    relative_level = max(expected_level, 0.0) / hist_mean
    return 0.6 + 0.4 * min(relative_level, 1.0)


def _historical_anomaly_frequency(historical_values: list[float], sigma: float) -> float:
    """跳跃超过3西格玛噪声乐队的历史步骤的小分数。

    高频表示度量 内在的噪音/跳动，所以异常
    更不要说惊恐了，最后的得分应该减少。
    """
    if len(historical_values) < 3 or sigma <= 0:
        return 0.0

    diffs = [historical_values[i] - historical_values[i - 1] for i in range(1, len(historical_values))]
    center = _median(diffs)
    # sigma是每步噪声;diff系列有std sigma* sqrt(2).
    threshold = SIGMA_BOUND * sigma * math.sqrt(2)
    if threshold <= 0:
        return 0.0
    violations = sum(1 for d in diffs if abs(d - center) > threshold)
    return violations / len(diffs)


def _evaluate_window(
    actual_values: list[float],
    predicted_values: list[float],
    historical_values: list[float],
    sigmas: list[float] | None = None,
    drop_weight: float = DEFAULT_DROP_WEIGHT,
    rise_weight: float = DEFAULT_RISE_WEIGHT,
    smoothing: float = DEFAULT_SMOOTHING,
) -> tuple[float, dict[str, Any]]:
    """评估最近点数的窗口并返回(得分在[0, 1],细节中)。

    窗口分数归结于它的 LAST 点(主播);前一个
    积分只提供光滑/遮蔽和持续时间的上下文
    证实。 ``smoothing`` 将主锚本身的贡献与 a 混合
    窗口最新加权(线性)平均值: 0 保持原始锚( 大部分)
    ),1使用全平滑平均值(最去诺化)。
    随后应用频率衰减和持续时长加权。
    """
    if not actual_values or not predicted_values:
        return 0.0, {"error": "Empty values provided for evaluation"}

    window_size = min(len(actual_values), len(predicted_values))

    # 每点西格玛没有时使用的全球噪声尺度。
    global_sigma = _estimate_noise_scale(historical_values)
    anomaly_frequency = _historical_anomaly_frequency(historical_values, global_sigma)
    frequency_dampener = 1.0 - 0.4 * min(anomaly_frequency, 1.0)

    # 平滑内核的线性重量:最古老的 -> 最新=1 -> window_size .
    weights = [float(i + 1) for i in range(window_size)]
    total_weight = sum(weights)
    normalized_weights = [w / total_weight for w in weights]

    contributions: list[float] = []
    point_details: list[dict[str, Any]] = []
    consecutive_anomalies = 0

    for i in range(window_size):
        actual = actual_values[i]
        predicted = predicted_values[i]
        weight = normalized_weights[i]
        sigma_i = sigmas[i] if sigmas is not None and i < len(sigmas) else global_sigma

        significance, z = _deviation_significance(actual, predicted, sigma_i)

        direction = "drop" if z < 0 else "rise"
        direction_weight = drop_weight if z < 0 else rise_weight
        severity = min(significance * direction_weight, 1.0)

        volume_factor = _volume_factor(predicted, historical_values)
        contribution = severity * volume_factor

        if severity > ANOMALY_THRESHOLD:
            consecutive_anomalies += 1
        else:
            consecutive_anomalies = 0

        contributions.append(contribution)

        point_details.append(
            {
                "actual": actual,
                "predicted": predicted,
                "z_score": z,
                "significance": significance,
                "direction": direction,
                "volume_factor": volume_factor,
                "severity": severity,
                "contribution": contribution,
                "weight": weight,
            }
        )

    # Anchor 归属: 分数属于最后一个点。前一个
    # 通过一个轻度的 最新加权平均值控制它
    # ``smoothing`` (轴重共享=1-平滑).
    anchor_contribution = contributions[-1]
    smoothed_contribution = sum(c * w for c, w in zip(contributions, normalized_weights, strict=False))
    smoothing = min(max(smoothing, 0.0), 1.0)
    window_base = (1.0 - smoothing) * anchor_contribution + smoothing * smoothed_contribution

    # 持续因子:连续数点异常时的助推。
    duration_boost = 0.0
    if consecutive_anomalies > 1:
        duration_boost = min((consecutive_anomalies - 1) * 0.1, 0.2)

    final_score = max(0.0, min(window_base * frequency_dampener + duration_boost, 1.0))

    details = {
        "window_size": window_size,
        "noise_scale": global_sigma,
        "historical_anomaly_frequency": anomaly_frequency,
        "frequency_dampener": frequency_dampener,
        "weighted_base": window_base,
        "anchor_contribution": anchor_contribution,
        "smoothed_contribution": smoothed_contribution,
        "smoothing": smoothing,
        "consecutive_anomalies": consecutive_anomalies,
        "duration_boost": duration_boost,
        "point_details": point_details,
    }

    return final_score, details


def evaluate_anomalies(
    input_data: list[float | int | dict[str, Any]],
    evaluation_window: int = 3,
    frequency: str = "hourly",
    target_column: str = "value",
    input_length: int | None = None,
    drop_weight: float = DEFAULT_DROP_WEIGHT,
    rise_weight: float = DEFAULT_RISE_WEIGHT,
    smoothing: float = DEFAULT_SMOOTHING,
) -> tuple[float, dict[str, Any]]:
    """对评价窗口进行预测，并对异常情况进行评分。

    ``evaluation_window`` 最后一个 ``input_data`` 点作为
    要评价的点数; 以上点数都用作预测输入
    作为噪音规模、音量和频率的历史背景
    因素。

    Args:
        input_data: 时间序列作为数字或dicts;评价点位于结尾。
        evaluation_window: 要评价的后端点数( 必须是< len( input_data))。
        frequency: 时间序列频率(如 'hourly' , 'daily').
        target_column: 要从结构化(dit)数据读取的列。
        input_length: 目标服务的历史输入长度;如果提供的历史是
            更短，服务左板的最早点数为0,以达到这个长度。
        drop_weight: 对向下偏差适用了严重性乘法。
        rise_weight: 对向上偏差适用了严重性乘法。

    Returns:
        Tuple(异常得分为[0, 1],细节为dict)。任何失败时，
        得分为0.0,细节包含一个 ``"error"`` 键。
    """
    if not input_data or len(input_data) <= evaluation_window:
        return 0.0, {
            "error": (
                f"Input data length ({len(input_data)}) must be greater than "
                f"evaluation window ({evaluation_window})."
            )
        }

    # 在进行任何工作之前检查服务提供情况。
    if not check_forecast_service_health():
        return 0.0, {"error": "Time Series Forecasting Service Unavailable. Cannot perform anomaly detection."}

    service_url = config.get().timeseries_forecasting_service_url

    # 将数据分为历史背景和评价窗口。
    historical_data = input_data[:-evaluation_window]
    evaluation_data = input_data[-evaluation_window:]

    hist_values = _extract_values(historical_data, target_column)
    eval_values = _extract_values(evaluation_data, target_column)

    if not hist_values or not eval_values:
        return 0.0, {"error": "Could not extract numeric values from input data."}

    # 仅利用历史背景预测评价窗口。
    forecast_result = call_timeseries_service(
        service_url=service_url,
        input_data=historical_data,
        forecast_window=evaluation_window,
        frequency=frequency,
        input_length=input_length,
        target_column=target_column,
    )

    if forecast_result.get("status") in ("error", "http_error"):
        return 0.0, {
            "error": f"Forecasting Error during anomaly detection: {forecast_result.get('error', 'Unknown error')}"
        }

    predicted_values = forecast_result.get("predictions", [])

    if not predicted_values or len(predicted_values) != len(eval_values):
        return 0.0, {
            "error": (
                f"Mismatch between predicted values ({len(predicted_values)}) "
                f"and evaluation window size ({len(eval_values)})."
            )
        }

    # 与前置兼容:如果预测服务返回到每个视距
    # 不确定性，比历史噪声尺度估计更可取。
    sigmas = forecast_result.get("prediction_std")
    if not isinstance(sigmas, list) or len(sigmas) != len(predicted_values):
        sigmas = None

    return _evaluate_window(
        actual_values=eval_values,
        predicted_values=predicted_values,
        historical_values=hist_values,
        sigmas=sigmas,
        drop_weight=drop_weight,
        rise_weight=rise_weight,
        smoothing=smoothing,
    )


def _block_forecast(
    input_data: list[float | int | dict[str, Any]],
    scan_start: int,
    detection_range: int,
    stride: int,
    frequency: str,
    input_length: int | None,
    target_column: str,
    service_url: str,
) -> tuple[list[float | None], list[dict[str, Any]]]:
    """在大小为 ``stride`` 的非重叠块中预测扫描范围。

    每个区块根据它之前的历史预测( ``input_data[:a]``),
    因此块内反常的实际值从不输入模型输入。
    返回长度 ``detection_range`` 的列表( 与扫描一致的预测)
    点; ``None`` 其中块失败)和失败块记录列表。
    """
    predicted: list[float | None] = [None] * detection_range
    failed_blocks: list[dict[str, Any]] = []

    offset = 0
    while offset < detection_range:
        anchor_abs = scan_start + offset
        horizon = min(stride, detection_range - offset)
        history = input_data[:anchor_abs]

        result = call_timeseries_service(
            service_url=service_url,
            input_data=history,
            forecast_window=horizon,
            frequency=frequency,
            input_length=input_length,
            target_column=target_column,
        )

        if result.get("status") in ("error", "http_error"):
            failed_blocks.append(
                {"start": offset, "end": offset + horizon - 1, "error": result.get("error", "Unknown error")}
            )
        else:
            preds = result.get("predictions", [])
            usable = min(len(preds), horizon)
            for k in range(usable):
                try:
                    predicted[offset + k] = float(preds[k])
                except (ValueError, TypeError):
                    predicted[offset + k] = None
            if usable < horizon:
                failed_blocks.append(
                    {
                        "start": offset + usable,
                        "end": offset + horizon - 1,
                        "error": f"Service returned only {len(preds)} of {horizon} predictions",
                    }
                )

        offset += horizon

    return predicted, failed_blocks


def _build_severity_curve(
    scan_actual: list[float],
    scan_predicted: list[float | None],
    historical_values: list[float],
    evaluation_window: int,
    drop_weight: float,
    rise_weight: float,
    smoothing: float,
) -> tuple[list[float], list[str]]:
    """将下方窗口滑过扫描范围并返回每个点的重度。

    每个锚点索引的后导窗口( 最多 ``evaluation_window`` 点)
    结束于主播)是得分;得分是主播的重度。
    窗口只从具有有效预测的毗连点建起，所以
    每点得分精确一次，失败的预测缺口被跳过。
    """
    n = len(scan_actual)
    severity = [0.0] * n
    directions = ["none"] * n

    for i in range(n):
        pred_i = scan_predicted[i]
        if pred_i is None:
            continue

        lo = max(0, i - evaluation_window + 1)
        window_actual: list[float] = []
        window_predicted: list[float] = []
        for j in range(lo, i + 1):
            pred_j = scan_predicted[j]
            if pred_j is None:
                # 重置，使窗口保持毗连并结束于锚上。
                window_actual = []
                window_predicted = []
                continue
            window_actual.append(scan_actual[j])
            window_predicted.append(pred_j)

        if not window_actual:
            continue

        score, _ = _evaluate_window(
            actual_values=window_actual,
            predicted_values=window_predicted,
            historical_values=historical_values,
            drop_weight=drop_weight,
            rise_weight=rise_weight,
            smoothing=smoothing,
        )
        severity[i] = score
        directions[i] = "drop" if scan_actual[i] < pred_i else "rise"

    return severity, directions


def _segment_intervals(severity: list[float], t_high: float, t_low: float) -> list[tuple[int, int]]:
    """Hysteresis 将重度曲线分解成 [起始，结束] 运行。

    当严重性升至/高于 ``t_high`` 并仅关闭时，间隔会打开
    当它跌落到 ``t_low`` ( ``t_low < t_high``) 下时，它会压制
    在一个门槛上闲聊
    """
    intervals: list[tuple[int, int]] = []
    start: int | None = None

    for i, s in enumerate(severity):
        if start is None:
            if s >= t_high:
                start = i
        elif s < t_low:
            intervals.append((start, i - 1))
            start = None

    if start is not None:
        intervals.append((start, len(severity) - 1))

    return intervals


def _merge_intervals(intervals: list[tuple[int, int]], gap: int) -> list[tuple[int, int]]:
    """以 ``gap`` 点分隔的间隔合并。"""
    if not intervals:
        return []
    merged = [intervals[0]]
    for cur_start, cur_end in intervals[1:]:
        prev_start, prev_end = merged[-1]
        if cur_start - prev_end - 1 <= gap:
            merged[-1] = (prev_start, cur_end)
        else:
            merged.append((cur_start, cur_end))
    return merged


def detect_anomaly_range(
    input_data: list[float | int | dict[str, Any]],
    detection_range: int,
    frequency: str = "hourly",
    target_column: str = "value",
    input_length: int | None = None,
    stride: int | None = None,
    evaluation_window: int | None = None,
    drop_weight: float = DEFAULT_DROP_WEIGHT,
    rise_weight: float = DEFAULT_RISE_WEIGHT,
    smoothing: float = DEFAULT_SMOOTHING,
    t_high: float = DEFAULT_T_HIGH,
    t_low: float = DEFAULT_T_LOW,
    merge_gap: int = DEFAULT_MERGE_GAP,
    area_min: float = DEFAULT_AREA_MIN,
    peak_high: float = DEFAULT_PEAK_HIGH,
) -> tuple[float, dict[str, Any]]:
    """扫描一个长的异常范围，然后将其归纳为异常间隔。

    后面的 ``detection_range`` 点被扫描;它们面前的一切都是
    用作预测/文本历史。范围以 ``stride`` 块预测
    (块预测),下方窗口滑过它产生一个每个点
    重度曲线，且曲线被分割(hysteresis)和过滤(区域 /)
    峰值)变为零或以上的异常间隔。

    Args:
        input_data: 连续无缺失的时间序列，扫描范围位于末尾。
        detection_range: 扫描的后导点数( 必须是< len( input_data))。
        frequency: 时间序列频率;驱动 ``evaluation_window`` 和 ``stride`` 默认值。
        target_column: 要从结构化(dit)数据读取的列。
        input_length: 每个区块预测服务的目标历史输入长度。
        stride: 屏蔽预测视野; ``frequency`` 当 None 时默认值。
        evaluation_window: 下窗大小; ``frequency`` 时默认为 None .
        drop_weight: 对向下偏差适用了严重性乘法。
        rise_weight: 对向上偏差适用了严重性乘法。
        smoothing: 下窗平滑强度在[0, 1].
        t_high: Hysteresis 上限打开间隔。
        t_low: Hysteresis 下限关闭间隔。
        merge_gap: 将间隔由 QQ 分隔这么多点。
        area_min: 最小间隔区域( 严重程度之和) 以保留它。
        peak_high: 山峰严重性，无论面积大小，都保持间隔。

    Returns:
        Tuple (总分为[0, 1] = 最大间隔峰值，细节 dict). On
        任何失败的分数为 0.0,细节包含一个 ``"error"`` 键。
    """
    if evaluation_window is None:
        evaluation_window = _default_evaluation_window(frequency)
    if stride is None:
        stride = _default_stride(frequency)

    if detection_range <= 0:
        return 0.0, {"error": f"detection_range ({detection_range}) must be positive."}
    if not input_data or len(input_data) <= detection_range:
        return 0.0, {
            "error": (
                f"Input data length ({len(input_data)}) must be greater than "
                f"detection_range ({detection_range}) to provide forecast history."
            )
        }

    if not check_forecast_service_health():
        return 0.0, {"error": "Time Series Forecasting Service Unavailable. Cannot perform anomaly detection."}

    service_url = config.get().timeseries_forecasting_service_url

    scan_start = len(input_data) - detection_range
    hist_values = _extract_values(input_data[:scan_start], target_column)
    scan_values = _extract_values(input_data[scan_start:], target_column)

    if not hist_values or len(scan_values) != detection_range:
        return 0.0, {"error": "Could not extract a continuous numeric series for the detection range."}

    predicted, failed_blocks = _block_forecast(
        input_data=input_data,
        scan_start=scan_start,
        detection_range=detection_range,
        stride=stride,
        frequency=frequency,
        input_length=input_length,
        target_column=target_column,
        service_url=service_url,
    )

    forecast_calls = math.ceil(detection_range / stride)

    if all(p is None for p in predicted):
        return 0.0, {
            "error": "All block forecasts failed during range anomaly detection.",
            "failed_blocks": failed_blocks,
        }

    severity, directions = _build_severity_curve(
        scan_actual=scan_values,
        scan_predicted=predicted,
        historical_values=hist_values,
        evaluation_window=evaluation_window,
        drop_weight=drop_weight,
        rise_weight=rise_weight,
        smoothing=smoothing,
    )

    raw_intervals = _segment_intervals(severity, t_high, t_low)
    merged_intervals = _merge_intervals(raw_intervals, merge_gap)

    intervals: list[dict[str, Any]] = []
    for seg_start, seg_end in merged_intervals:
        seg_severity = severity[seg_start : seg_end + 1]
        area = sum(seg_severity)
        peak = max(seg_severity)
        if area < area_min and peak < peak_high:
            continue

        # 最重要的方向是 积累最严重的方向。
        drop_mass = sum(severity[i] for i in range(seg_start, seg_end + 1) if directions[i] == "drop")
        rise_mass = sum(severity[i] for i in range(seg_start, seg_end + 1) if directions[i] == "rise")
        direction = "drop" if drop_mass >= rise_mass else "rise"

        start_abs = scan_start + seg_start
        end_abs = scan_start + seg_end
        peak_offset = seg_start + seg_severity.index(peak)

        intervals.append(
            {
                "start_index": seg_start,
                "end_index": seg_end,
                "start_abs_index": start_abs,
                "end_abs_index": end_abs,
                "start_timestamp": _extract_timestamp(input_data[start_abs]),
                "end_timestamp": _extract_timestamp(input_data[end_abs]),
                "length": seg_end - seg_start + 1,
                "peak": peak,
                "peak_index": peak_offset,
                "peak_timestamp": _extract_timestamp(input_data[scan_start + peak_offset]),
                "mean": area / len(seg_severity),
                "area": area,
                "direction": direction,
                "points": [
                    {
                        "index": i,
                        "actual": scan_values[i],
                        "predicted": predicted[i],
                        "severity": severity[i],
                        "direction": directions[i],
                    }
                    for i in range(seg_start, seg_end + 1)
                ],
            }
        )

    overall_score = max((iv["peak"] for iv in intervals), default=0.0)

    details = {
        "detection_range": detection_range,
        "frequency": frequency,
        "stride": stride,
        "evaluation_window": evaluation_window,
        "forecast_calls": forecast_calls,
        "scan_start_index": scan_start,
        "severity_curve": severity,
        "intervals": intervals,
        "failed_blocks": failed_blocks,
    }

    return overall_score, details


def format_anomaly_report(score: float, details: dict[str, Any]) -> str:
    """将异常检测结果格式化为人可读报告。"""
    if "error" in details:
        return f"Anomaly Detection Error: {details['error']}"

    severity = "Low"
    if score > 0.8:
        severity = "Critical"
    elif score > 0.6:
        severity = "High"
    elif score > 0.4:
        severity = "Medium"

    report = [
        "🚨 Anomaly Detection Report",
        f"Overall Anomaly Score: {score:.4f} (Severity: {severity})",
        "",
        "Evaluation Summary:",
        f"  • Window Size: {details.get('window_size', 0)} points",
        f"  • Noise Scale (sigma): {details.get('noise_scale', 0.0):.4f}",
        f"  • Historical Anomaly Frequency: {details.get('historical_anomaly_frequency', 0.0):.4f}",
        f"  • Frequency Dampener: x{details.get('frequency_dampener', 1.0):.4f}",
        f"  • Consecutive Anomalies: {details.get('consecutive_anomalies', 0)}",
        f"  • Duration Boost Applied: +{details.get('duration_boost', 0.0):.4f}",
        "",
        "Detailed Point Evaluation (Oldest to Newest):",
    ]

    for i, point in enumerate(details.get("point_details", [])):
        report.extend(
            [
                f"  Point {i + 1} (Weight: {point['weight']:.2f}):",
                f"    - Actual: {point['actual']:.2f} | Predicted: {point['predicted']:.2f} " f"({point['direction']})",
                f"    - Z-Score: {point['z_score']:.2f} | Significance: {point['significance']:.4f}",
                f"    - Volume Factor: x{point['volume_factor']:.4f}",
                f"    - Point Severity: {point['severity']:.4f} | Contribution: {point['contribution']:.4f}",
            ]
        )

    if score > 0.6:
        report.extend(
            [
                "",
                "💡 Insight: Significant anomaly detected. The actual values deviate "
                "substantially from the expected pattern.",
            ]
        )
    else:
        report.extend(
            [
                "",
                "💡 Insight: No significant anomaly detected. The data follows the expected pattern closely.",
            ]
        )

    return "\n".join(report)


def _severity_label(score: float) -> str:
    """将异常分数映射到粗重的标签上。"""
    if score > 0.8:
        return "Critical"
    if score > 0.6:
        return "High"
    if score > 0.4:
        return "Medium"
    return "Low"


def format_anomaly_range_report(score: float, details: dict[str, Any]) -> str:
    """将一个范围(滑动)异常检测结果格式化为可读报告。"""
    if "error" in details:
        return f"Anomaly Detection Error: {details['error']}"

    intervals = details.get("intervals", [])
    failed_blocks = details.get("failed_blocks", [])

    report = [
        "🚨 Anomaly Range Detection Report",
        f"Overall Anomaly Score: {score:.4f} (Severity: {_severity_label(score)})",
        "",
        "Scan Summary:",
        f"  • Detection Range: {details.get('detection_range', 0)} points ({details.get('frequency', 'unknown')})",
        f"  • Stride (forecast horizon): {details.get('stride', 0)} | "
        f"Evaluation Window: {details.get('evaluation_window', 0)} | "
        f"Forecast Calls: {details.get('forecast_calls', 0)}",
        f"  • Anomalous Intervals Found: {len(intervals)}",
    ]

    if failed_blocks:
        report.append(
            f"  • ⚠️ Degraded: {len(failed_blocks)} forecast block(s) failed and were skipped "
            "(those points were not scored)."
        )

    report.append("")

    if not intervals:
        report.append("💡 Insight: No significant anomalous interval detected across the scanned range.")
        return "\n".join(report)

    for idx, iv in enumerate(intervals):
        start_label = iv.get("start_timestamp")
        end_label = iv.get("end_timestamp")
        if start_label is not None and end_label is not None:
            span = f"{start_label} → {end_label}"
        else:
            span = f"index {iv['start_index']}..{iv['end_index']}"
        report.extend(
            [
                f"Interval {idx + 1} [{span}] ({iv['direction']}):",
                f"    - Length: {iv['length']} points | Peak: {iv['peak']:.4f} "
                f"(Severity: {_severity_label(iv['peak'])}) | Mean: {iv['mean']:.4f} | Area: {iv['area']:.4f}",
            ]
        )

    report.extend(
        [
            "",
            f"💡 Insight: {len(intervals)} anomalous interval(s) detected. The most severe peaks "
            f"to a score of {score:.4f}.",
        ]
    )

    return "\n".join(report)
