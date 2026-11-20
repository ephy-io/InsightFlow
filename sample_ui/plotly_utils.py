"""Plotly 从可视化 DSL 生成图表的公用设备。"""

from io import StringIO
from typing import Any

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


def create_plotly_chart(data_csv: str, visualization_dsl: dict[str, Any]) -> go.Figure:
    """从 CSV 数据和可视化 DSL 创建图。

    Args:
        data_csv: 包含数据的 CSV 字符串
        visualization_dsl: 包含图表配置的词典

    Returns:
        Plotly 图表对象
    """
    if not data_csv or not visualization_dsl:
        return create_empty_chart("No data available")

    if "error" in visualization_dsl:
        return create_empty_chart(f"Visualization error: {visualization_dsl['error']}")

    try:
        # 解析 CSV 数据
        df = pd.read_csv(StringIO(data_csv))

        if df.empty:
            return create_empty_chart("No data to visualize")

        chart_type = visualization_dsl.get("chart_type", "table")
        config = visualization_dsl.get("config", {})
        layout = visualization_dsl.get("layout", {})

        # 基于类型创建图表
        if chart_type == "line":
            return create_line_chart(df, config, layout)
        elif chart_type == "bar":
            return create_bar_chart(df, config, layout)
        elif chart_type == "pie":
            return create_pie_chart(df, config, layout)
        elif chart_type == "scatter":
            return create_scatter_chart(df, config, layout)
        elif chart_type == "histogram":
            return create_histogram_chart(df, config, layout)
        elif chart_type == "box":
            return create_box_chart(df, config, layout)
        elif chart_type == "table":
            return create_table_chart(df, config, layout)
        else:
            return create_empty_chart(f"Unsupported chart type: {chart_type}")

    except Exception as e:
        return create_empty_chart(f"Chart generation error: {str(e)}")


def create_line_chart(df: pd.DataFrame, config: dict[str, Any], layout: dict[str, Any]) -> go.Figure:
    """创建行图。"""
    x_col = config.get("x")
    y_col = config.get("y")
    color_col = config.get("color")

    if not x_col or x_col not in df.columns:
        return create_empty_chart("Missing required x column for line chart")

    # 处理多个 Y 列大小写
    if isinstance(y_col, list):
        # 多种衡量标准 -- -- 需要融化数据
        if not all(col in df.columns for col in y_col):
            return create_empty_chart("Some y columns missing from data")

        # 将数据框熔化为多个序列的长格式
        melted_df = df.melt(id_vars=[x_col], value_vars=y_col, var_name="metric", value_name="value")
        fig = px.line(melted_df, x=x_col, y="value", color="metric")

    else:
        # 单 Y 栏
        if not y_col or y_col not in df.columns:
            return create_empty_chart("Missing required y column for line chart")

        # 检查颜色列是否存在且是否有效
        if color_col and color_col in df.columns:
            fig = px.line(df, x=x_col, y=y_col, color=color_col)
        else:
            fig = px.line(df, x=x_col, y=y_col)

    fig.update_layout(**layout)
    return fig


def create_bar_chart(df: pd.DataFrame, config: dict[str, Any], layout: dict[str, Any]) -> go.Figure:
    """创建柱状图。"""
    x_col = config.get("x")
    y_col = config.get("y")

    if not x_col or x_col not in df.columns:
        return create_empty_chart("Missing required x column for bar chart")

    # 处理多个 Y 列大小写
    if isinstance(y_col, list):
        # 多种衡量标准 -- -- 需要融化数据
        if not all(col in df.columns for col in y_col):
            return create_empty_chart("Some y columns missing from data")

        # 将数据框熔化为多个序列的长格式
        melted_df = df.melt(id_vars=[x_col], value_vars=y_col, var_name="metric", value_name="value")
        fig = px.bar(melted_df, x=x_col, y="value", color="metric")

    else:
        # 单 Y 栏
        if not y_col or y_col not in df.columns:
            return create_empty_chart("Missing required y column for bar chart")

        fig = px.bar(df, x=x_col, y=y_col)

    fig.update_layout(**layout)
    return fig


def create_pie_chart(df: pd.DataFrame, config: dict[str, Any], layout: dict[str, Any]) -> go.Figure:
    """创建馅饼图。"""
    labels_col = config.get("labels")
    values_col = config.get("values")

    if not labels_col or not values_col or labels_col not in df.columns or values_col not in df.columns:
        return create_empty_chart("Missing required columns for pie chart")

    fig = px.pie(df, names=labels_col, values=values_col)
    fig.update_layout(**layout)
    return fig


def create_scatter_chart(df: pd.DataFrame, config: dict[str, Any], layout: dict[str, Any]) -> go.Figure:
    """创建一个散开的模式。"""
    x_col = config.get("x")
    y_col = config.get("y")

    if not x_col or not y_col or x_col not in df.columns or y_col not in df.columns:
        return create_empty_chart("Missing required columns for scatter plot")

    fig = px.scatter(df, x=x_col, y=y_col)
    fig.update_layout(**layout)
    return fig


def create_histogram_chart(df: pd.DataFrame, config: dict[str, Any], layout: dict[str, Any]) -> go.Figure:
    """创建直方图。"""
    x_col = config.get("x")
    nbins = config.get("nbins", 20)

    if not x_col or x_col not in df.columns:
        return create_empty_chart("Missing required column for histogram")

    fig = px.histogram(df, x=x_col, nbins=nbins)
    fig.update_layout(**layout)
    return fig


def create_box_chart(df: pd.DataFrame, config: dict[str, Any], layout: dict[str, Any]) -> go.Figure:
    """创建框图。"""
    y_col = config.get("y")
    x_col = config.get("x")

    if not y_col or y_col not in df.columns:
        return create_empty_chart("Missing required column for box plot")

    if x_col and x_col in df.columns:
        fig = px.box(df, x=x_col, y=y_col)
    else:
        fig = px.box(df, y=y_col)

    fig.update_layout(**layout)
    return fig


def create_table_chart(df: pd.DataFrame, config: dict[str, Any], layout: dict[str, Any]) -> go.Figure:
    """创建表格显示。"""
    columns = config.get("columns", list(df.columns))

    # 限制为显示的前100行
    display_df = df.head(100)

    fig = go.Figure(
        data=[
            go.Table(
                header={"values": columns, "fill_color": "lightblue", "align": "left"},
                cells={
                    "values": [display_df[col] for col in columns if col in display_df.columns],
                    "fill_color": "white",
                    "align": "left",
                },
            )
        ]
    )

    fig.update_layout(**layout)
    return fig


def create_empty_chart(message: str) -> go.Figure:
    """创建带有消息的空图表。"""
    fig = go.Figure()
    fig.add_annotation(
        text=message,
        xref="paper",
        yref="paper",
        x=0.5,
        y=0.5,
        xanchor="center",
        yanchor="middle",
        showarrow=False,
        font={"size": 16},
    )
    fig.update_layout(
        title="Chart Generation Issue",
        xaxis={"showgrid": False, "showticklabels": False, "zeroline": False},
        yaxis={"showgrid": False, "showticklabels": False, "zeroline": False},
    )
    return fig


def visualization_dsl_to_gradio_plot(data_csv: str, visualization_dsl: dict[str, Any]) -> tuple[go.Figure, str]:
    """将可视化的 DSL 转换为 Gradio - 相容的图。

    Args:
        data_csv: 包含数据的 CSV 字符串
        visualization_dsl: 包含图表配置的词典

    Returns:
        图解( 拼图，描述字符串)
    """
    fig = create_plotly_chart(data_csv, visualization_dsl)

    if visualization_dsl:
        chart_type = visualization_dsl.get("chart_type", "unknown")
        layout_title = visualization_dsl.get("layout", {}).get("title", f"{chart_type.title()} Chart")
        description = f"Generated {chart_type} visualization: {layout_title}"
    else:
        description = "Data table view"

    return fig, description


def create_inline_chart_markdown(data_csv: str, visualization_dsl: dict[str, Any]) -> str:
    """为内含显示创建图表的简化缩放表示。

    这样可以创建基于文本的概要，并有可点击的链接来显示交互图表。
    """
    if not data_csv or not visualization_dsl:
        return "📊 *No visualization data available*"

    if "error" in visualization_dsl:
        return f"⚠️ *Visualization error: {visualization_dsl['error']}*"

    try:
        from io import StringIO

        import pandas as pd

        df = pd.read_csv(StringIO(data_csv))
        chart_type = visualization_dsl.get("chart_type", "table")
        layout = visualization_dsl.get("layout", {})
        title = layout.get("title", f"{chart_type.title()} Chart")

        # 创建带有关键数据点和视图指示的文本摘要
        summary_lines = [
            f"📊 **{title}**",
            "",
            f"*Chart Type: {chart_type.title()}* | *Data Points: {len(df)} rows, {len(df.columns)} columns*",
            "",
            "✨ **Interactive chart will appear automatically in the chart panel →**",
            "",
        ]

        # 添加一个小数据样本
        if len(df) > 0:
            summary_lines.append("**Sample Data:**")
            summary_lines.append("```")
            # 以干净的格式显示前几行
            sample_df = df.head(3)
            summary_lines.append(sample_df.to_string(index=False))
            if len(df) > 3:
                summary_lines.append(f"... and {len(df) - 3} more rows")
            summary_lines.append("```\n")

        return "\n".join(summary_lines)

    except Exception as e:
        return f"⚠️ *Chart generation error: {str(e)}*"
