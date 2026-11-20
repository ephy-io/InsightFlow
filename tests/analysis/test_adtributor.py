import pandas as pd

from insightflow.analysis.adtributor import adtributor
from insightflow.tool.adtributor_tool import adtributor_drilldown


def test_adtributor_absolute():
    # 设置类似演示的测试数据
    df_site = pd.DataFrame(
        {
            "site_section": [1, 2, 3],
            "predict": [0.5, 0.1, 0.9],
            "real": [0.7, 0.9, 1.0],
            "proportion": [0.2, 0.5, 0.3],
            "base_proportion": [0.2, 0.5, 0.3],
        }
    )
    df_dict = {"site_section": df_site}

    # 执行
    output = adtributor(derived=False, df_dict=df_dict, issue_type="drop", tep=0.7)

    # 注:关于下降，我们看预测> 真实之处。
    # 在演示数据中：
    # 第1行:预测0.5,实际0.7 -> 预测< 真实(上升)
    # 第2行:预测0.1,实际0.9 -> 预测< 真实(上升)
    # 第3行:预测0.9,实数1.0 -> 预测< 实数(上升)
    # None 他们下降!
    assert output.status == "no_anomaly_direction"

    # 来试一下真正的滴水
    df_site_drop = pd.DataFrame(
        {
            "site_section": [1, 2, 3],
            "predict": [10.0, 5.0, 2.0],
            "real": [8.0, 1.0, 2.0],  # 1和2个下降，2个严重下降
            "proportion": [0.5, 0.3, 0.2],
            "base_proportion": [0.5, 0.3, 0.2],
        }
    )
    output = adtributor(derived=False, df_dict={"site_section": df_site_drop}, issue_type="drop", tep=0.7)

    assert output.status == "success"
    assert "site_section" in output.root_causes
    # 2个从5个降至1个(与规模相比大幅下降),1个从10个降至8个
    # 2应该是主要根源
    assert 2 in output.root_causes["site_section"]


BASE_FIELDS = {
    "predict": None,
    "real": None,
    "predict_numerator": None,
    "predict_denominator": None,
    "real_numerator": None,
    "real_denominator": None,
    "proportion": None,
    "base_proportion": None,
}


def test_adtributor_tool_interface():
    # 测试已熔化的表格格式
    melted_data = [
        {**BASE_FIELDS, "dimension_name": "device", "element_value": "ios", "predict": 1000, "real": 500},
        {**BASE_FIELDS, "dimension_name": "device", "element_value": "android", "predict": 2000, "real": 1900},
        {**BASE_FIELDS, "dimension_name": "province", "element_value": "guangdong", "predict": 500, "real": 200},
        {**BASE_FIELDS, "dimension_name": "province", "element_value": "beijing", "predict": 500, "real": 450},
    ]

    result = adtributor_drilldown.invoke(
        {"reasoning": "Test drilldown", "data": melted_data, "derived": False, "issue_type": "drop", "k": 2}
    )

    assert "error" not in result
    assert result["status"] == "success"
    assert "device" in result["dimension_details"]
    assert "province" in result["dimension_details"]

    # iOS下降了500(50%),而roid下降了100(5%). iOS应该是一个根源。
    assert "ios" in result["root_causes"].get("device", [])
    assert "guangdong" in result["root_causes"].get("province", [])

    # 描述性检查
    narrative = result["dimension_details"]["device"]["narrative"]
    assert "ios" in narrative
    assert "contributed to" in narrative


def test_adtributor_derived_metric_tool():
    # 通过工具接口测试衍生的度量
    melted_data = [
        {
            **BASE_FIELDS,
            "dimension_name": "ad_type",
            "element_value": "banner",
            "predict_numerator": 50,  # 点击
            "predict_denominator": 1000,  # 印象 -> CTR 5%
            "real_numerator": 10,
            "real_denominator": 1000,  # CTR 1% -> 重降
        },
        {
            **BASE_FIELDS,
            "dimension_name": "ad_type",
            "element_value": "video",
            "predict_numerator": 20,
            "predict_denominator": 100,  # CTR 20%
            "real_numerator": 19,
            "real_denominator": 100,  # CTR 19% -> 轻微下降
        },
    ]

    result = adtributor_drilldown.invoke(
        {"reasoning": "Test derived drilldown", "data": melted_data, "derived": True, "issue_type": "drop"}
    )

    assert "error" not in result
    assert result["status"] == "success"
    assert "banner" in result["root_causes"].get("ad_type", [])


def test_adtributor_complex_case():
    import random

    random.seed(42)

    dimensions = ["region", "device", "browser", "channel"]
    melted_data = []

    # 每个维数生成12个属性(至少 10)
    for dim in dimensions:
        for i in range(12):
            attr_name = f"{dim}_{i}"
            # 基本价值
            predict_val = random.randint(500, 1500)

            # 注射异常: Region_3 和 Device_7 有重滴。
            if attr_name == "region_3":
                real_val = predict_val * 0.1  # 下降90%
            elif attr_name == "device_7":
                real_val = predict_val * 0.15  # 85%的下降率
            else:
                # 正常波动(+-5%)
                real_val = predict_val * random.uniform(0.95, 1.05)

            melted_data.append(
                {
                    **BASE_FIELDS,
                    "dimension_name": dim,
                    "element_value": attr_name,
                    "predict": predict_val,
                    "real": real_val,
                }
            )

    result = adtributor_drilldown.invoke(
        {
            "reasoning": "Test complex drilldown with 4 dims and 10+ attrs",
            "data": melted_data,
            "derived": False,
            "issue_type": "drop",
            "k": 2,
        }
    )

    assert "error" not in result
    assert result["status"] == "success"

    # 检查那些有重滴的药被检测为根本原因
    assert "region_3" in result["root_causes"].get("region", [])
    assert "device_7" in result["root_causes"].get("device", [])

    # 校验叙述存在并包含根源
    assert "region_3" in result["dimension_details"]["region"]["narrative"]
    assert "device_7" in result["dimension_details"]["device"]["narrative"]
