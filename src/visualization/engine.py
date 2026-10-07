"""Data Visualization Engine for interactive financial dashboards.

Provides chart creation, dashboard assembly, and HTML report generation
using Plotly and Dash.
"""
from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


@dataclass
class ChartConfig:
    """Configuration for individual charts."""

    title: str = ""
    width: int = 800
    height: int = 600
    show_legend: bool = True
    template: str = "plotly"


@dataclass
class DashboardConfig:
    """Configuration for dashboards."""

    title: str = "Dashboard"
    columns: int = 2
    theme: str = "plotly"
    template: str = "plotly"


class VisualizationEngine:
    """Engine for creating interactive financial visualizations."""

    def __init__(
        self,
        theme: str = "plotly",
        default_width: int = 800,
        default_height: int = 600,
    ):
        self.theme = theme
        self.default_width = default_width
        self.default_height = default_height
        self._charts: dict[str, go.Figure] = {}
        self._dashboards: dict[str, Any] = {}
        self._color_palette: list[str] = []

    # ------------------------------------------------------------------
    # Theme & Styling
    # ------------------------------------------------------------------

    def set_theme(self, theme: str) -> None:
        """Set the default theme for all charts."""
        self.theme = theme

    def set_color_palette(self, colors: list[str]) -> None:
        """Set a custom color palette."""
        self._color_palette = colors

    def _get_template(self, config: Optional[ChartConfig] = None) -> str:
        """Resolve the template to use."""
        if config and config.template:
            return config.template
        return self.theme

    def _get_dimensions(
        self, config: Optional[ChartConfig] = None
    ) -> tuple[int, int]:
        """Resolve width and height."""
        if config:
            return config.width, config.height
        return self.default_width, self.default_height

    def _store_chart(self, chart: go.Figure, chart_id: Optional[str]) -> None:
        """Store a chart in the internal registry."""
        if chart_id:
            self._charts[chart_id] = chart

    # ------------------------------------------------------------------
    # Chart Creation
    # ------------------------------------------------------------------

    def create_line_chart(
        self,
        data: pd.DataFrame,
        x: str,
        y: str,
        title: str = "",
        config: Optional[ChartConfig] = None,
        chart_id: Optional[str] = None,
    ) -> go.Figure:
        """Create a line chart."""
        if data.empty:
            raise ValueError("Cannot create chart from empty DataFrame")
        if x not in data.columns or y not in data.columns:
            raise KeyError(f"Column not found: {x} or {y}")

        template = self._get_template(config)
        width, height = self._get_dimensions(config)
        show_legend = config.show_legend if config else True

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=data[x],
                y=data[y],
                mode="lines",
                name=y,
                line=dict(color=self._color_palette[0] if self._color_palette else None),
            )
        )
        fig.update_layout(
            title=title,
            width=width,
            height=height,
            template=template,
            showlegend=show_legend,
        )
        self._store_chart(fig, chart_id)
        return fig

    def create_multi_line_chart(
        self,
        data: pd.DataFrame,
        x: str,
        y_columns: list[str],
        title: str = "",
        config: Optional[ChartConfig] = None,
        chart_id: Optional[str] = None,
    ) -> go.Figure:
        """Create a multi-line chart with multiple y-series."""
        if data.empty:
            raise ValueError("Cannot create chart from empty DataFrame")
        if x not in data.columns:
            raise KeyError(f"Column not found: {x}")
        for col in y_columns:
            if col not in data.columns:
                raise KeyError(f"Column not found: {col}")

        template = self._get_template(config)
        width, height = self._get_dimensions(config)
        show_legend = config.show_legend if config else True

        fig = go.Figure()
        for i, col in enumerate(y_columns):
            color = (
                self._color_palette[i]
                if self._color_palette and i < len(self._color_palette)
                else None
            )
            fig.add_trace(
                go.Scatter(
                    x=data[x],
                    y=data[col],
                    mode="lines",
                    name=col,
                    line=dict(color=color),
                )
            )
        fig.update_layout(
            title=title,
            width=width,
            height=height,
            template=template,
            showlegend=show_legend,
        )
        self._store_chart(fig, chart_id)
        return fig

    def create_bar_chart(
        self,
        data: pd.DataFrame,
        x: str,
        y: str,
        title: str = "",
        orientation: str = "v",
        marker_color: Optional[list[str]] = None,
        config: Optional[ChartConfig] = None,
        chart_id: Optional[str] = None,
    ) -> go.Figure:
        """Create a bar chart."""
        if data.empty:
            raise ValueError("Cannot create chart from empty DataFrame")
        if x not in data.columns or y not in data.columns:
            raise KeyError(f"Column not found: {x} or {y}")

        template = self._get_template(config)
        width, height = self._get_dimensions(config)
        show_legend = config.show_legend if config else True

        fig = go.Figure()
        fig.add_trace(
            go.Bar(
                x=data[x] if orientation == "v" else data[y],
                y=data[y] if orientation == "v" else data[x],
                orientation=orientation,
                marker_color=marker_color,
                name=y,
            )
        )
        fig.update_layout(
            title=title,
            width=width,
            height=height,
            template=template,
            showlegend=show_legend,
        )
        self._store_chart(fig, chart_id)
        return fig

    def create_candlestick_chart(
        self,
        data: pd.DataFrame,
        x: str,
        open_col: str,
        high_col: str,
        low_col: str,
        close_col: str,
        volume_col: Optional[str] = None,
        title: str = "",
        config: Optional[ChartConfig] = None,
        chart_id: Optional[str] = None,
    ) -> go.Figure:
        """Create a candlestick chart with optional volume subplot."""
        if data.empty:
            raise ValueError("Cannot create chart from empty DataFrame")
        required = [x, open_col, high_col, low_col, close_col]
        for col in required:
            if col not in data.columns:
                raise KeyError(f"Column not found: {col}")

        template = self._get_template(config)
        width, height = self._get_dimensions(config)
        show_legend = config.show_legend if config else True

        rows = 2 if volume_col else 1
        row_heights = [0.7, 0.3] if volume_col else [1.0]

        fig = make_subplots(
            rows=rows,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.03,
            row_heights=row_heights,
        )

        fig.add_trace(
            go.Candlestick(
                x=data[x],
                open=data[open_col],
                high=data[high_col],
                low=data[low_col],
                close=data[close_col],
                name="OHLC",
            ),
            row=1,
            col=1,
        )

        if volume_col:
            colors = [
                "green" if data[close_col].iloc[i] >= data[open_col].iloc[i] else "red"
                for i in range(len(data))
            ]
            fig.add_trace(
                go.Bar(
                    x=data[x],
                    y=data[volume_col],
                    marker_color=colors,
                    name="Volume",
                    showlegend=False,
                ),
                row=2,
                col=1,
            )

        fig.update_layout(
            title=title,
            width=width,
            height=height,
            template=template,
            showlegend=show_legend,
            xaxis_rangeslider_visible=False,
        )
        self._store_chart(fig, chart_id)
        return fig

    def create_scatter_chart(
        self,
        data: pd.DataFrame,
        x: str,
        y: str,
        title: str = "",
        trendline: Optional[str] = None,
        config: Optional[ChartConfig] = None,
        chart_id: Optional[str] = None,
    ) -> go.Figure:
        """Create a scatter chart with optional trendline."""
        if data.empty:
            raise ValueError("Cannot create chart from empty DataFrame")
        if x not in data.columns or y not in data.columns:
            raise KeyError(f"Column not found: {x} or {y}")

        template = self._get_template(config)
        width, height = self._get_dimensions(config)
        show_legend = config.show_legend if config else True

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=data[x],
                y=data[y],
                mode="markers",
                name=f"{x} vs {y}",
            )
        )

        if trendline == "ols":
            x_vals = np.array(data[x], dtype=float)
            y_vals = np.array(data[y], dtype=float)
            mask = ~(np.isnan(x_vals) | np.isnan(y_vals))
            if mask.sum() > 1:
                coeffs = np.polyfit(x_vals[mask], y_vals[mask], 1)
                trend_y = np.polyval(coeffs, x_vals[mask])
                fig.add_trace(
                    go.Scatter(
                        x=x_vals[mask],
                        y=trend_y,
                        mode="lines",
                        name="Trend",
                        line=dict(dash="dash", color="red"),
                    )
                )

        fig.update_layout(
            title=title,
            width=width,
            height=height,
            template=template,
            showlegend=show_legend,
        )
        self._store_chart(fig, chart_id)
        return fig

    def create_pie_chart(
        self,
        data: pd.DataFrame,
        names: str,
        values: str,
        title: str = "",
        config: Optional[ChartConfig] = None,
        chart_id: Optional[str] = None,
    ) -> go.Figure:
        """Create a pie chart."""
        if data.empty:
            raise ValueError("Cannot create chart from empty DataFrame")
        if names not in data.columns or values not in data.columns:
            raise KeyError(f"Column not found: {names} or {values}")

        template = self._get_template(config)
        width, height = self._get_dimensions(config)

        fig = go.Figure()
        fig.add_trace(
            go.Pie(
                labels=data[names],
                values=data[values],
                hole=0.3,
            )
        )
        fig.update_layout(
            title=title,
            width=width,
            height=height,
            template=template,
        )
        self._store_chart(fig, chart_id)
        return fig

    def create_correlation_heatmap(
        self,
        data: pd.DataFrame,
        columns: list[str],
        title: str = "",
        config: Optional[ChartConfig] = None,
        chart_id: Optional[str] = None,
    ) -> go.Figure:
        """Create a correlation heatmap."""
        if data.empty:
            raise ValueError("Cannot create chart from empty DataFrame")
        for col in columns:
            if col not in data.columns:
                raise KeyError(f"Column not found: {col}")

        template = self._get_template(config)
        width, height = self._get_dimensions(config)

        corr = data[columns].corr()

        fig = go.Figure()
        fig.add_trace(
            go.Heatmap(
                z=corr.values,
                x=columns,
                y=columns,
                colorscale="RdBu",
                zmid=0,
                text=np.round(corr.values, 2),
                texttemplate="%{text}",
                textfont={"size": 10},
            )
        )
        fig.update_layout(
            title=title,
            width=width,
            height=height,
            template=template,
        )
        self._store_chart(fig, chart_id)
        return fig

    # ------------------------------------------------------------------
    # Financial Metrics Charts
    # ------------------------------------------------------------------

    def create_drawdown_chart(
        self,
        data: pd.DataFrame,
        price_col: str,
        title: str = "",
        config: Optional[ChartConfig] = None,
        chart_id: Optional[str] = None,
    ) -> go.Figure:
        """Create a drawdown chart from price data."""
        if data.empty:
            raise ValueError("Cannot create chart from empty DataFrame")
        if price_col not in data.columns:
            raise KeyError(f"Column not found: {price_col}")

        template = self._get_template(config)
        width, height = self._get_dimensions(config)

        prices = data[price_col]
        cumulative_max = prices.cummax()
        drawdown = (prices - cumulative_max) / cumulative_max

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=data.index if "date" not in data.columns else data["date"],
                y=drawdown,
                mode="lines",
                fill="tozeroy",
                name="Drawdown",
                line=dict(color="red"),
            )
        )
        fig.update_layout(
            title=title or "Drawdown",
            width=width,
            height=height,
            template=template,
            yaxis_title="Drawdown",
            yaxis_tickformat=".1%",
        )
        self._store_chart(fig, chart_id)
        return fig

    def create_rolling_returns_chart(
        self,
        data: pd.DataFrame,
        price_col: str,
        window: int = 20,
        title: str = "",
        config: Optional[ChartConfig] = None,
        chart_id: Optional[str] = None,
    ) -> go.Figure:
        """Create a rolling returns chart."""
        if data.empty:
            raise ValueError("Cannot create chart from empty DataFrame")
        if price_col not in data.columns:
            raise KeyError(f"Column not found: {price_col}")

        template = self._get_template(config)
        width, height = self._get_dimensions(config)

        prices = data[price_col]
        rolling_returns = prices.pct_change(periods=window)

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=data.index if "date" not in data.columns else data["date"],
                y=rolling_returns,
                mode="lines",
                name=f"{window}-period Return",
                line=dict(color="blue"),
            )
        )
        fig.update_layout(
            title=title or f"Rolling {window}-Period Returns",
            width=width,
            height=height,
            template=template,
            yaxis_title="Return",
            yaxis_tickformat=".1%",
        )
        self._store_chart(fig, chart_id)
        return fig

    def create_cumulative_returns_chart(
        self,
        data: pd.DataFrame,
        columns: list[str],
        title: str = "",
        config: Optional[ChartConfig] = None,
        chart_id: Optional[str] = None,
    ) -> go.Figure:
        """Create cumulative returns chart for multiple assets."""
        if data.empty:
            raise ValueError("Cannot create chart from empty DataFrame")
        for col in columns:
            if col not in data.columns:
                raise KeyError(f"Column not found: {col}")

        template = self._get_template(config)
        width, height = self._get_dimensions(config)

        fig = go.Figure()
        x_vals = data["date"] if "date" in data.columns else data.index
        for col in columns:
            cumulative = (1 + data[col]).cumprod() - 1
            fig.add_trace(
                go.Scatter(
                    x=x_vals,
                    y=cumulative,
                    mode="lines",
                    name=col,
                )
            )
        fig.update_layout(
            title=title or "Cumulative Returns",
            width=width,
            height=height,
            template=template,
            yaxis_title="Cumulative Return",
            yaxis_tickformat=".1%",
        )
        self._store_chart(fig, chart_id)
        return fig

    # ------------------------------------------------------------------
    # Dashboard Creation
    # ------------------------------------------------------------------

    def create_dashboard(
        self,
        charts: list[dict[str, Any]],
        config: Optional[DashboardConfig] = None,
        chart_id: Optional[str] = None,
    ) -> dict[str, Any]:
        """Create a dashboard from chart specifications."""
        if config is None:
            config = DashboardConfig()

        dashboard_charts = []
        for spec in charts:
            chart_type = spec.get("type", "")
            if chart_type == "line":
                fig = self.create_line_chart(
                    data=spec["data"],
                    x=spec["x"],
                    y=spec["y"],
                    title=spec.get("title", ""),
                )
            elif chart_type == "bar":
                fig = self.create_bar_chart(
                    data=spec["data"],
                    x=spec["x"],
                    y=spec["y"],
                    title=spec.get("title", ""),
                )
            elif chart_type == "candlestick":
                fig = self.create_candlestick_chart(
                    data=spec["data"],
                    x=spec["x"],
                    open_col=spec["open"],
                    high_col=spec["high"],
                    low_col=spec["low"],
                    close_col=spec["close"],
                    volume_col=spec.get("volume"),
                    title=spec.get("title", ""),
                )
            elif chart_type == "scatter":
                fig = self.create_scatter_chart(
                    data=spec["data"],
                    x=spec["x"],
                    y=spec["y"],
                    title=spec.get("title", ""),
                )
            elif chart_type == "pie":
                fig = self.create_pie_chart(
                    data=spec["data"],
                    names=spec["names"],
                    values=spec["values"],
                    title=spec.get("title", ""),
                )
            elif chart_type == "heatmap":
                fig = self.create_correlation_heatmap(
                    data=spec["data"],
                    columns=spec["columns"],
                    title=spec.get("title", ""),
                )
            else:
                raise ValueError(f"Unsupported chart type: {chart_type}")

            dashboard_charts.append(fig)

        dashboard = {
            "title": config.title,
            "columns": config.columns,
            "theme": config.theme,
            "charts": dashboard_charts,
        }

        if chart_id:
            self._dashboards[chart_id] = dashboard

        return dashboard

    def create_dashboard_from_dict(
        self,
        chart_specs: list[dict[str, Any]],
        title: str = "Dashboard",
    ) -> dict[str, Any]:
        """Create a dashboard from a list of chart specification dicts."""
        config = DashboardConfig(title=title)
        return self.create_dashboard(charts=chart_specs, config=config)

    # ------------------------------------------------------------------
    # Report Generation
    # ------------------------------------------------------------------

    def generate_html_report(
        self,
        charts: list[go.Figure],
        title: str = "Report",
        output_path: Optional[str] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> str:
        """Generate an HTML report from a list of charts."""
        if output_path is None:
            fd, output_path = tempfile.mkstemp(suffix=".html", prefix="viz_report_")
            os.close(fd)

        html_parts = [
            "<!DOCTYPE html>",
            "<html>",
            "<head>",
            f"<title>{title}</title>",
            '<script src="https://cdn.plot.ly/plotly-latest.min.js"></script>',
            "<style>",
            "body { font-family: Arial, sans-serif; margin: 20px; }",
            ".chart { margin: 20px 0; }",
            "h1 { color: #333; }",
            ".metadata { color: #666; margin-bottom: 20px; }",
            "</style>",
            "</head>",
            "<body>",
            f"<h1>{title}</h1>",
        ]

        if metadata:
            html_parts.append('<div class="metadata">')
            for key, value in metadata.items():
                html_parts.append(f"<p><strong>{key}:</strong> {value}</p>")
            html_parts.append("</div>")

        for i, chart in enumerate(charts):
            chart_html = chart.to_html(full_html=False, include_plotlyjs=False)
            html_parts.append(f'<div class="chart" id="chart_{i}">')
            html_parts.append(chart_html)
            html_parts.append("</div>")

        html_parts.extend(["</body>", "</html>"])

        with open(output_path, "w", encoding="utf-8") as f:
            f.write("\n".join(html_parts))

        return output_path

    # ------------------------------------------------------------------
    # Chart Export
    # ------------------------------------------------------------------

    def export_chart(
        self,
        chart: go.Figure,
        output_path: str,
        format: str = "html",
    ) -> None:
        """Export a chart to a file."""
        if format == "html":
            chart.write_html(output_path)
        elif format == "json":
            chart_json = chart.to_json()
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(json.loads(chart_json), f, indent=2)
        elif format == "png":
            chart.write_image(output_path)
        else:
            raise ValueError(f"Unsupported export format: {format}")

    # ------------------------------------------------------------------
    # Chart Retrieval
    # ------------------------------------------------------------------

    def get_chart(self, chart_id: str) -> Optional[go.Figure]:
        """Retrieve a chart by ID."""
        return self._charts.get(chart_id)

    def list_charts(self) -> list[str]:
        """List all stored chart IDs."""
        return list(self._charts.keys())

    def clear_charts(self) -> None:
        """Clear all stored charts."""
        self._charts.clear()
