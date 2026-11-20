import pytest

from insightflow.analysis import anomaly_detection as ad
from insightflow.analysis.anomaly_detection import (
    _default_evaluation_window,
    _default_stride,
    _deviation_significance,
    _estimate_noise_scale,
    _evaluate_window,
    _extract_values,
    _historical_anomaly_frequency,
    _merge_intervals,
    _segment_intervals,
    _volume_factor,
    detect_anomaly_range,
    evaluate_anomalies,
    format_anomaly_range_report,
    format_anomaly_report,
)

# 平滑，轻度吵闹的历史 与中~1000。
SMOOTH_HISTORY = [1000.0, 1010.0, 990.0, 1005.0, 995.0, 1002.0, 998.0, 1008.0, 992.0, 1004.0] * 3
# 服务路径测试所使用的历史较长。
LONG_HISTORY = SMOOTH_HISTORY * 4


def test_extract_values():
    assert _extract_values([1.0, 2.0, 3.0]) == [1.0, 2.0, 3.0]
    assert _extract_values([1, 2, 3]) == [1.0, 2.0, 3.0]
    assert _extract_values([{"value": 10}, {"value": 20}]) == [10.0, 20.0]
    assert _extract_values([{"custom": 10}, {"custom": 20}], target_column="custom") == [10.0, 20.0]

    # 边界案件
    assert _extract_values([]) == []
    assert _extract_values([{"wrong_col": 10}]) == []
    assert _extract_values([{"value": "not_a_number"}, {"value": 20}]) == [20.0]
    assert _extract_values([None, "string", 10.0]) == [10.0]
    # 布尔不能作为数字数据处理
    assert _extract_values([True, False, 5]) == [5.0]


def test_estimate_noise_scale():
    # 恒定系列没有噪音。
    assert _estimate_noise_scale([100.0, 100.0, 100.0]) == 0.0
    # 估计太短了
    assert _estimate_noise_scale([]) == 0.0
    assert _estimate_noise_scale([100.0]) == 0.0
    # 具有变异性的系列产生正比分。
    assert _estimate_noise_scale(SMOOTH_HISTORY) > 0.0


def test_deviation_significance():
    # 无偏差。
    sig, z = _deviation_significance(100.0, 100.0, 10.0)
    assert sig == 0.0
    assert z == 0.0
    # 恰好在三西格玛线上 -> 0.5。
    sig, z = _deviation_significance(130.0, 100.0, 10.0)
    assert sig == pytest.approx(0.5)
    assert z == pytest.approx(3.0)
    # 6 sigma -> 饱和度为1.0。
    sig, _ = _deviation_significance(160.0, 100.0, 10.0)
    assert sig == pytest.approx(1.0)
    # 方向通过z的标志来捕捉。
    _, z_down = _deviation_significance(70.0, 100.0, 10.0)
    assert z_down < 0
    # 零西格玛是地板上的，所以仍然探测到一个平面系列上的清晰跳跃。
    sig, _ = _deviation_significance(101.0, 100.0, 0.0)
    assert sig > 0.5


def test_volume_factor():
    # 空历史 -> 中性。
    assert _volume_factor(1000.0, []) == 1.0
    # 非阳性平均值 -> 中性。
    assert _volume_factor(1000.0, [0.0, 0.0]) == 1.0
    # 高于历史平均值的级别 -> 全部重量。
    assert _volume_factor(1500.0, SMOOTH_HISTORY) == pytest.approx(1.0)
    # 低流量的瞬间被挡在0.6
    assert _volume_factor(0.0, SMOOTH_HISTORY) == pytest.approx(0.6)
    # 中层介于两个极端之间。
    assert 0.6 < _volume_factor(500.0, SMOOTH_HISTORY) < 0.9


def test_historical_anomaly_frequency():
    # 平滑历史 -> ~没有历史异常。
    assert _historical_anomaly_frequency(SMOOTH_HISTORY, _estimate_noise_scale(SMOOTH_HISTORY)) < 0.1
    # 降伏诸投入。
    assert _historical_anomaly_frequency([1.0, 2.0], 1.0) == 0.0
    assert _historical_anomaly_frequency(SMOOTH_HISTORY, 0.0) == 0.0


def test_historical_anomaly_frequency_detects_spikes():
    # 大多是光滑的系列，有一些孤立的尖锐:强烈的噪声尺度
    # 保持小点，所以尖锐 被计算为历史异常。
    spiky = [1000.0 + (5.0 if i % 2 else -5.0) for i in range(30)]
    for idx in (7, 15, 23):
        spiky[idx] = 1600.0
    freq = _historical_anomaly_frequency(spiky, _estimate_noise_scale(spiky))
    assert freq > 0.0


def test_evaluate_window_normal():
    actual = [1000.0, 1010.0, 990.0]
    predicted = [1000.0, 1000.0, 1000.0]
    score, details = _evaluate_window(actual, predicted, SMOOTH_HISTORY)
    assert score < 0.3
    assert details["consecutive_anomalies"] == 0


def test_evaluate_window_legit_growth_not_flagged():
    # 较大但预期值(实际匹配预测)必须标注 NOT ,
    # 尽管它的绝对规模很大。
    actual = [2000.0, 2000.0, 2000.0]
    predicted = [2000.0, 2000.0, 2000.0]
    score, _ = _evaluate_window(actual, predicted, SMOOTH_HISTORY)
    assert score < 0.1


def test_evaluate_window_drop_to_zero_high_traffic():
    # 高流量测量值降到零是最严重的异常
    # 必须得分接近1(相对大小的标志错误的返回测试)。
    actual = [1000.0, 500.0, 0.0]
    predicted = [1000.0, 1000.0, 1000.0]
    score, details = _evaluate_window(actual, predicted, SMOOTH_HISTORY)
    assert score > 0.8
    assert details["point_details"][-1]["direction"] == "drop"


def test_evaluate_window_direction_weighting():
    # 使用轻度偏差，在线性(<3 sigma)波段内保持，所以
    # 方向重量不为饱和所掩盖. SMOOTH_HISTORY sigma~10.
    predicted = [1000.0, 1000.0, 1000.0]
    drop_actual = [1000.0, 1000.0, 985.0]
    rise_actual = [1000.0, 1000.0, 1015.0]

    drop_default, _ = _evaluate_window(drop_actual, predicted, SMOOTH_HISTORY)
    drop_emphasised, _ = _evaluate_window(drop_actual, predicted, SMOOTH_HISTORY, drop_weight=2.0, rise_weight=1.0)
    rise_default, _ = _evaluate_window(rise_actual, predicted, SMOOTH_HISTORY)

    # 突出的下降会提高向下偏差的得分。
    assert drop_emphasised > drop_default
    # 以中性重量计算，对称的下降和上升得分相同。
    assert drop_default == pytest.approx(rise_default)


def test_evaluate_window_noisy_history_dampens():
    # 同样的绝对偏差在高变异性上远没有那么显著
    # 高噪声序列上的异常得分应低于平滑序列。
    actual = [1000.0, 1000.0, 1500.0]
    predicted = [1000.0, 1000.0, 1000.0]
    noisy_history = [1000.0, 100.0, 1900.0, 200.0, 1800.0, 50.0, 1950.0, 150.0] * 3

    smooth_score, _ = _evaluate_window(actual, predicted, SMOOTH_HISTORY)
    noisy_score, _ = _evaluate_window(actual, predicted, noisy_history)
    assert noisy_score < smooth_score


def test_evaluate_window_duration_boost():
    actual = [400.0, 350.0, 300.0]
    predicted = [1000.0, 1000.0, 1000.0]
    score, details = _evaluate_window(actual, predicted, SMOOTH_HISTORY)
    assert details["consecutive_anomalies"] == 3
    assert details["duration_boost"] > 0.0
    assert score > 0.8


def test_evaluate_window_boundary_cases():
    # 空输入。
    score, details = _evaluate_window([], [], SMOOTH_HISTORY)
    assert score == 0.0
    assert "error" in details
    # 匹配的长度 -> 仅评价重叠。
    _, details = _evaluate_window([100.0], [100.0, 100.0], SMOOTH_HISTORY)
    assert details["window_size"] == 1
    # 每点西格玛在供应时会得到荣誉。
    score, _ = _evaluate_window([100.0], [100.0], SMOOTH_HISTORY, sigmas=[1.0])
    assert score == 0.0


def _patch_service(monkeypatch, predictions, status="success", healthy=True):
    """补齐 evaluate_anomalies 使用的预测服务依赖性。"""
    monkeypatch.setattr(ad, "check_forecast_service_health", lambda: healthy)

    class _FakeCfg:
        timeseries_forecasting_service_url = "http://fake"

    monkeypatch.setattr(ad.config, "get", lambda: _FakeCfg())

    def _fake_call(**kwargs):
        if status != "success":
            return {"status": status, "error": "boom"}
        return {"predictions": predictions, "status": "success"}

    monkeypatch.setattr(ad, "call_timeseries_service", _fake_call)


def test_evaluate_anomalies_success(monkeypatch):
    data = LONG_HISTORY + [1000.0, 500.0, 0.0]
    _patch_service(monkeypatch, predictions=[1000.0, 1000.0, 1000.0])
    score, details = evaluate_anomalies(data, evaluation_window=3)
    assert "error" not in details
    assert score > 0.8


def test_evaluate_anomalies_input_too_short():
    score, details = evaluate_anomalies([1.0, 2.0], evaluation_window=3)
    assert score == 0.0
    assert "error" in details


def test_evaluate_anomalies_short_history_proceeds(monkeypatch):
    # 短历史必须 NOT 硬- error; 最小模式的粘贴被授权
    # 传递给预测服务 call_timeseries_service，以便继续检测。
    _patch_service(monkeypatch, predictions=[1.0, 2.0, 3.0])
    short = SMOOTH_HISTORY + [1.0, 2.0, 3.0]  # 30个历史< 模型最小值
    score, details = evaluate_anomalies(short, evaluation_window=3)
    assert "error" not in details


def test_evaluate_anomalies_service_unavailable(monkeypatch):
    monkeypatch.setattr(ad, "check_forecast_service_health", lambda: False)
    score, details = evaluate_anomalies(LONG_HISTORY + [1.0, 2.0, 3.0], evaluation_window=3)
    assert score == 0.0
    assert "Unavailable" in details["error"]


def test_evaluate_anomalies_forecast_error(monkeypatch):
    _patch_service(monkeypatch, predictions=[], status="error")
    score, details = evaluate_anomalies(LONG_HISTORY + [1.0, 2.0, 3.0], evaluation_window=3)
    assert score == 0.0
    assert "error" in details


def test_evaluate_anomalies_prediction_length_mismatch(monkeypatch):
    _patch_service(monkeypatch, predictions=[1000.0])  # 仅 1, 需要 3
    score, details = evaluate_anomalies(LONG_HISTORY + [1.0, 2.0, 3.0], evaluation_window=3)
    assert score == 0.0
    assert "Mismatch" in details["error"]


def test_format_anomaly_report():
    score, details = _evaluate_window([1000.0, 500.0, 0.0], [1000.0, 1000.0, 1000.0], SMOOTH_HISTORY)
    report = format_anomaly_report(score, details)
    assert "Anomaly Detection Report" in report
    assert "Severity" in report
    # 错误细节导出错误报告。
    assert "Anomaly Detection Error" in format_anomaly_report(0.0, {"error": "boom"})


# ---------------------------------------------------------------------------
# 下窗主机归属和平滑
# ---------------------------------------------------------------------------


def test_evaluate_window_anchor_attribution():
    # 只有锚(最后一点)偏离;得分必须反映。
    predicted = [1000.0, 1000.0, 1000.0]
    anchor_only = [1000.0, 1000.0, 940.0]
    score, details = _evaluate_window(anchor_only, predicted, SMOOTH_HISTORY)
    assert score > 0.0
    assert details["point_details"][-1]["direction"] == "drop"


def test_evaluate_window_smoothing_strength():
    # 独锚比无平滑(纯锚)得分高
    # 完全平滑，其中前两个正常点将比分拉低。
    predicted = [1000.0, 1000.0, 1000.0]
    spike = [1000.0, 1000.0, 940.0]
    sharp, _ = _evaluate_window(spike, predicted, SMOOTH_HISTORY, smoothing=0.0)
    smoothed, _ = _evaluate_window(spike, predicted, SMOOTH_HISTORY, smoothing=1.0)
    assert sharp > smoothed


# ---------------------------------------------------------------------------
# 频率驱动默认值
# ---------------------------------------------------------------------------


def test_default_evaluation_window():
    assert _default_evaluation_window("10min") == 6
    assert _default_evaluation_window("minutely") == 6
    assert _default_evaluation_window("hourly") == 3
    assert _default_evaluation_window("daily") == 3
    assert _default_evaluation_window("weekly") == 3
    assert _default_evaluation_window("") == 3


def test_default_stride():
    assert _default_stride("5min") == 12
    assert _default_stride("hourly") == 12
    assert _default_stride("daily") == 7
    assert _default_stride("weekly") == 4
    assert _default_stride("monthly") == 4
    assert _default_stride("quarterly") == 10  # 倒转


# ---------------------------------------------------------------------------
# Hysteresis 分割和合并
# ---------------------------------------------------------------------------


def test_segment_intervals_hysteresis():
    # 达到 t_high (0.6) 时开启，低于 t_low (0.4) 时结束。
    severity = [0.0, 0.7, 0.5, 0.45, 0.2, 0.0, 0.9, 0.1]
    intervals = _segment_intervals(severity, t_high=0.6, t_low=0.4)
    assert intervals == [(1, 3), (6, 6)]


def test_segment_intervals_open_until_end():
    severity = [0.1, 0.7, 0.8]
    assert _segment_intervals(severity, t_high=0.6, t_low=0.4) == [(1, 2)]


def test_merge_intervals():
    # 1点(指数3)的差值在1,2和4,5之间---->当差值========= 1.
    assert _merge_intervals([(1, 2), (4, 5)], gap=1) == [(1, 5)]
    # 更大的分离是分开的。
    assert _merge_intervals([(1, 2), (6, 7)], gap=1) == [(1, 2), (6, 7)]
    assert _merge_intervals([], gap=1) == []


# ---------------------------------------------------------------------------
# 上层: detect_anomaly_range
# ---------------------------------------------------------------------------


def _patch_block_service(monkeypatch, predict_fn, healthy=True):
    """补丁服务，所以每个块预测由 ``predict_fn`` 计算。

    ``predict_fn(history, horizon)`` 返回预测值列表。
    """
    monkeypatch.setattr(ad, "check_forecast_service_health", lambda: healthy)

    class _FakeCfg:
        timeseries_forecasting_service_url = "http://fake"

    monkeypatch.setattr(ad.config, "get", lambda: _FakeCfg())

    calls = {"count": 0}

    def _fake_call(**kwargs):
        calls["count"] += 1
        history = kwargs["input_data"]
        horizon = kwargs["forecast_window"]
        return {"predictions": predict_fn(history, horizon), "status": "success"}

    monkeypatch.setattr(ad, "call_timeseries_service", _fake_call)
    return calls


def test_detect_anomaly_range_finds_interval(monkeypatch):
    # 平面历史为1000;扫描范围在中间有明显的持续下降。
    history = [1000.0] * 60
    scan = [1000.0] * 24
    for i in range(8, 12):
        scan[i] = 100.0  # 持续下降
    data = history + scan

    # 预测模型预测每个区块的平坦基线1000.
    calls = _patch_block_service(monkeypatch, lambda hist, horizon: [1000.0] * horizon)

    score, details = detect_anomaly_range(data, detection_range=24, frequency="hourly", stride=12)
    assert "error" not in details
    assert score > 0.6
    assert len(details["intervals"]) >= 1
    # 检测到的间隔应覆盖注入的空投区域。
    iv = details["intervals"][0]
    assert iv["direction"] == "drop"
    assert iv["start_index"] <= 11 and iv["end_index"] >= 8
    # 按块预测，ceil(24/12) = 2 次调用。
    assert calls["count"] == 2
    assert details["forecast_calls"] == 2


def test_detect_anomaly_range_normal_no_intervals(monkeypatch):
    # 历史中带有现实的轻度噪音(西格玛~10),因此在其中的轻度噪音也非常相似。
    # 扫描范围停留在~1西格玛左右，没有标记。
    history = SMOOTH_HISTORY * 4  # 120分，平均值~1000,西格玛~10
    scan = [1000.0, 1010.0, 990.0, 1005.0, 995.0, 1002.0] * 4  # 24个温和点
    data = history + scan
    _patch_block_service(monkeypatch, lambda hist, horizon: [1000.0] * horizon)

    score, details = detect_anomaly_range(data, detection_range=24, frequency="hourly", stride=12)
    assert "error" not in details
    assert details["intervals"] == []
    assert score == 0.0


def test_detect_anomaly_range_multiple_intervals(monkeypatch):
    history = [1000.0] * 60
    scan = [1000.0] * 30
    for i in range(3, 6):
        scan[i] = 50.0  # 第一个异常( 投放)
    for i in range(20, 23):
        scan[i] = 2000.0  # 第二个异常点( 升)
    data = history + scan
    _patch_block_service(monkeypatch, lambda hist, horizon: [1000.0] * horizon)

    score, details = detect_anomaly_range(data, detection_range=30, frequency="hourly", stride=10)
    assert len(details["intervals"]) >= 2
    directions = {iv["direction"] for iv in details["intervals"]}
    assert "drop" in directions and "rise" in directions


def test_detect_anomaly_range_input_too_short():
    score, details = detect_anomaly_range([1.0, 2.0, 3.0], detection_range=5)
    assert score == 0.0
    assert "error" in details


def test_detect_anomaly_range_service_unavailable(monkeypatch):
    monkeypatch.setattr(ad, "check_forecast_service_health", lambda: False)
    score, details = detect_anomaly_range([1000.0] * 80, detection_range=24)
    assert score == 0.0
    assert "Unavailable" in details["error"]


def test_detect_anomaly_range_all_blocks_fail(monkeypatch):
    monkeypatch.setattr(ad, "check_forecast_service_health", lambda: True)

    class _FakeCfg:
        timeseries_forecasting_service_url = "http://fake"

    monkeypatch.setattr(ad.config, "get", lambda: _FakeCfg())
    monkeypatch.setattr(ad, "call_timeseries_service", lambda **k: {"status": "error", "error": "boom"})

    score, details = detect_anomaly_range([1000.0] * 80, detection_range=24, stride=12)
    assert score == 0.0
    assert "All block forecasts failed" in details["error"]


def test_detect_anomaly_range_partial_block_failure(monkeypatch):
    # 第一块失败，第二块成功 -> 已退化，但仍返回结果。
    monkeypatch.setattr(ad, "check_forecast_service_health", lambda: True)

    class _FakeCfg:
        timeseries_forecasting_service_url = "http://fake"

    monkeypatch.setattr(ad.config, "get", lambda: _FakeCfg())

    state = {"first": True}

    def _fake_call(**kwargs):
        if state["first"]:
            state["first"] = False
            return {"status": "error", "error": "boom"}
        return {"predictions": [1000.0] * kwargs["forecast_window"], "status": "success"}

    monkeypatch.setattr(ad, "call_timeseries_service", _fake_call)

    data = [1000.0] * 60 + [1000.0] * 24
    score, details = detect_anomaly_range(data, detection_range=24, frequency="hourly", stride=12)
    assert "error" not in details
    assert len(details["failed_blocks"]) == 1


def test_format_anomaly_range_report(monkeypatch):
    history = [1000.0] * 60
    scan = [1000.0] * 24
    for i in range(8, 12):
        scan[i] = 100.0
    data = history + scan
    _patch_block_service(monkeypatch, lambda hist, horizon: [1000.0] * horizon)

    score, details = detect_anomaly_range(data, detection_range=24, frequency="hourly", stride=12)
    report = format_anomaly_range_report(score, details)
    assert "Anomaly Range Detection Report" in report
    assert "Interval 1" in report
    # 错误路径。
    assert "Anomaly Detection Error" in format_anomaly_range_report(0.0, {"error": "boom"})
