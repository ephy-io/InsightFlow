from typing import Any

from pydantic import BaseModel, Field


class DimensionResult(BaseModel):
    explanatory_power: float | None = Field(default=None, description="该维度累计解释力")
    total_surprise: float = Field(description="该维度所有元素的惊异度总和")
    elements: list[Any] | None = Field(default=None, description="通过阈值筛选的根因元素")
    surprise: float | None = Field(default=None, description="根因元素的惊异度总和")
    reason: str = Field(description="选择或跳过该维度的原因")


class AdtributorOutput(BaseModel):
    root_causes: dict[str, list[Any]] = Field(description="按维度名称映射到根因元素列表")
    ranked_dimensions: list[str] = Field(
        description="按惊异度总和从高到低排列的已分析维度"
    )
    dimension_details: dict[str, DimensionResult] = Field(description="各维度的详细归因信息")
    status: str = Field(description="分析状态：success、no_root_cause 或 no_anomaly_direction")
    reason_flag: str = Field(default="", description="标记跳过分析的内部原因")
