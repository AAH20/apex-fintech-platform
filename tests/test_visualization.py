"""Tests for the Data Visualization Engine."""
import json
import os
import tempfile

import pandas as pd
import plotly.graph_objects as go
import pytest

from src.visualization.engine import VisualizationEngine, ChartConfig, DashboardConfig


@pytest.fixture
def engine():
    """Create a fresh VisualizationEngine instance."""
    return VisualizationEngine()


@pytest.fixture
def sample_ohlc_data():
    """Sample OHLC price data."""
    return pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=10, freq="D"),
            "open": [100, 102, 101, 103, 105, 104, 106, 108, 107, 109],
            "high": [103, 104, 103, 106, 107, 106, 109, 110, 109, 111],
            "low": [99, 100, 99, 101, 103, 102, 104, 106, 105, 107],
            "close": [102, 103, 102, 105, 106, 105, 108, 109, 108, 110],
            "volume": [1000, 1200, 900, 1500, 1800, 1100, 2000, 2200, 1300, 1600],
        }
    )


@pytest.fixture
def sample_returns_data():
    """Sample returns data for multiple assets."""
    return pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=5, freq="D"),
            "AAPL": [0.01, -0.005, 0.02, 0.015, -0.01],
            "GOOGL": [0.005, 0.01, -0.008, 0.012, 0.003],
            "MSFT": [-0.002, 0.008, 0.015, -0.005, 0.01],
        }
    )


@pytest.fixture
def sample_bar_data():
    """Sample categorical data for bar charts."""
    return pd.DataFrame(
        {
            "category": ["Revenue", "COGS", "Gross Profit", "OpEx", "Net Income"],
            "value": [1000, 600, 400, 250, 150],
        }
    )


class TestVisualizationEngineInit:
    """Test engine initialization."""

    def test_engine_initializes_with_defaults(self, engine):
        assert engine.theme == "plotly"
        assert engine.default_width == 800
        assert engine.default_height == 600
        assert engine._charts == {}
        assert engine._dashboards == {}

    def test_engine_initializes_with_custom_theme(self):
        eng = VisualizationEngine(theme="plotly_dark")
        assert eng.theme == "plotly_dark"

    def test_engine_initializes_with_custom_dimensions(self):
        eng = VisualizationEngine(default_width=1200, default_height=800)
        assert eng.default_width == 1200
        assert eng.default_height == 800


class TestChartConfig:
    """Test ChartConfig dataclass."""

    def test_chart_config_defaults(self):
        config = ChartConfig()
        assert config.title == ""
        assert config.width == 800
        assert config.height == 600
        assert config.show_legend is True
        assert config.template == "plotly"

    def test_chart_config_custom_values(self):
        config = ChartConfig(
            title="Test Chart", width=1000, height=500, show_legend=False, template="plotly_dark"
        )
        assert config.title == "Test Chart"
        assert config.width == 1000
        assert config.height == 500
        assert config.show_legend is False
        assert config.template == "plotly_dark"


class TestDashboardConfig:
    """Test DashboardConfig dataclass."""

    def test_dashboard_config_defaults(self):
        config = DashboardConfig()
        assert config.title == "Dashboard"
        assert config.columns == 2
        assert config.theme == "plotly"

    def test_dashboard_config_custom_values(self):
        config = DashboardConfig(title="My Dashboard", columns=3, template="plotly_dark")
        assert config.title == "My Dashboard"
        assert config.columns == 3
        assert config.template == "plotly_dark"


class TestLineChart:
    """Test line chart creation."""

    def test_create_line_chart(self, engine, sample_ohlc_data):
        chart = engine.create_line_chart(
            data=sample_ohlc_data,
            x="date",
            y="close",
            title="Stock Price",
        )
        assert isinstance(chart, go.Figure)
        assert chart.layout.title.text == "Stock Price"
        assert len(chart.data) == 1
        assert chart.data[0].type == "scatter"
        assert chart.data[0].mode == "lines"

    def test_create_line_chart_with_config(self, engine, sample_ohlc_data):
        config = ChartConfig(title="Custom Line", width=1000, height=400)
        chart = engine.create_line_chart(
            data=sample_ohlc_data, x="date", y="close", config=config
        )
        assert chart.layout.width == 1000
        assert chart.layout.height == 400

    def test_create_multi_line_chart(self, engine, sample_returns_data):
        chart = engine.create_multi_line_chart(
            data=sample_returns_data,
            x="date",
            y_columns=["AAPL", "GOOGL", "MSFT"],
            title="Returns Comparison",
        )
        assert isinstance(chart, go.Figure)
        assert len(chart.data) == 3
        assert chart.data[0].name == "AAPL"
        assert chart.data[1].name == "GOOGL"
        assert chart.data[2].name == "MSFT"

    def test_line_chart_stores_in_engine(self, engine, sample_ohlc_data):
        engine.create_line_chart(
            data=sample_ohlc_data, x="date", y="close", title="Test", chart_id="line1"
        )
        assert "line1" in engine._charts


class TestBarChart:
    """Test bar chart creation."""

    def test_create_bar_chart(self, engine, sample_bar_data):
        chart = engine.create_bar_chart(
            data=sample_bar_data,
            x="category",
            y="value",
            title="Financial Breakdown",
        )
        assert isinstance(chart, go.Figure)
        assert chart.data[0].type == "bar"
        assert chart.layout.title.text == "Financial Breakdown"

    def test_create_horizontal_bar_chart(self, engine, sample_bar_data):
        chart = engine.create_bar_chart(
            data=sample_bar_data,
            x="value",
            y="category",
            orientation="h",
            title="Horizontal Bars",
        )
        assert chart.data[0].orientation == "h"

    def test_bar_chart_with_colors(self, engine, sample_bar_data):
        colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
        chart = engine.create_bar_chart(
            data=sample_bar_data,
            x="category",
            y="value",
            marker_color=colors,
        )
        assert list(chart.data[0].marker.color) == colors


class TestCandlestickChart:
    """Test candlestick chart creation."""

    def test_create_candlestick_chart(self, engine, sample_ohlc_data):
        chart = engine.create_candlestick_chart(
            data=sample_ohlc_data,
            x="date",
            open_col="open",
            high_col="high",
            low_col="low",
            close_col="close",
            title="OHLC Candlestick",
        )
        assert isinstance(chart, go.Figure)
        assert chart.data[0].type == "candlestick"
        assert chart.layout.title.text == "OHLC Candlestick"

    def test_candlestick_with_volume(self, engine, sample_ohlc_data):
        chart = engine.create_candlestick_chart(
            data=sample_ohlc_data,
            x="date",
            open_col="open",
            high_col="high",
            low_col="low",
            close_col="close",
            volume_col="volume",
            title="Price + Volume",
        )
        # Should have candlestick + volume bar
        assert len(chart.data) == 2
        assert chart.data[0].type == "candlestick"
        assert chart.data[1].type == "bar"


class TestScatterChart:
    """Test scatter chart creation."""

    def test_create_scatter_chart(self, engine, sample_ohlc_data):
        chart = engine.create_scatter_chart(
            data=sample_ohlc_data,
            x="volume",
            y="close",
            title="Volume vs Price",
        )
        assert isinstance(chart, go.Figure)
        assert chart.data[0].type == "scatter"
        assert chart.data[0].mode == "markers"

    def test_scatter_with_trendline(self, engine, sample_ohlc_data):
        chart = engine.create_scatter_chart(
            data=sample_ohlc_data,
            x="volume",
            y="close",
            trendline="ols",
        )
        # Trendline adds a second trace
        assert len(chart.data) >= 2


class TestPieChart:
    """Test pie chart creation."""

    def test_create_pie_chart(self, engine, sample_bar_data):
        chart = engine.create_pie_chart(
            data=sample_bar_data,
            names="category",
            values="value",
            title="Revenue Distribution",
        )
        assert isinstance(chart, go.Figure)
        assert chart.data[0].type == "pie"
        assert chart.layout.title.text == "Revenue Distribution"


class TestHeatmapChart:
    """Test heatmap chart creation."""

    def test_create_correlation_heatmap(self, engine, sample_returns_data):
        chart = engine.create_correlation_heatmap(
            data=sample_returns_data,
            columns=["AAPL", "GOOGL", "MSFT"],
            title="Correlation Matrix",
        )
        assert isinstance(chart, go.Figure)
        assert chart.data[0].type == "heatmap"
        assert chart.layout.title.text == "Correlation Matrix"

    def test_correlation_heatmap_values(self, engine, sample_returns_data):
        chart = engine.create_correlation_heatmap(
            data=sample_returns_data, columns=["AAPL", "GOOGL", "MSFT"]
        )
        z_values = chart.data[0].z
        # Diagonal should be 1.0
        assert z_values[0][0] == pytest.approx(1.0, abs=0.01)
        assert z_values[1][1] == pytest.approx(1.0, abs=0.01)
        assert z_values[2][2] == pytest.approx(1.0, abs=0.01)


class TestDashboardCreation:
    """Test dashboard creation."""

    def test_create_dashboard(self, engine, sample_ohlc_data, sample_bar_data):
        charts = [
            {"type": "line", "data": sample_ohlc_data, "x": "date", "y": "close", "title": "Price"},
            {"type": "bar", "data": sample_bar_data, "x": "category", "y": "value", "title": "Breakdown"},
        ]
        dashboard = engine.create_dashboard(
            charts=charts, config=DashboardConfig(title="Test Dashboard", columns=2)
        )
        assert dashboard is not None
        assert "Test Dashboard" in str(dashboard)

    def test_dashboard_stores_in_engine(self, engine, sample_ohlc_data):
        charts = [
            {"type": "line", "data": sample_ohlc_data, "x": "date", "y": "close", "title": "Price"},
        ]
        engine.create_dashboard(
            charts=charts, config=DashboardConfig(title="Stored Dash"), chart_id="dash1"
        )
        assert "dash1" in engine._dashboards

    def test_create_dashboard_from_dict(self, engine, sample_ohlc_data):
        chart_specs = [
            {"type": "line", "data": sample_ohlc_data, "x": "date", "y": "close"},
        ]
        dashboard = engine.create_dashboard_from_dict(
            chart_specs=chart_specs, title="Dict Dashboard"
        )
        assert dashboard is not None


class TestReportGeneration:
    """Test report generation."""

    def test_generate_html_report(self, engine, sample_ohlc_data, sample_bar_data):
        charts = [
            engine.create_line_chart(sample_ohlc_data, "date", "close", title="Price Chart"),
            engine.create_bar_chart(sample_bar_data, "category", "value", title="Breakdown"),
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            report_path = os.path.join(tmpdir, "report.html")
            result = engine.generate_html_report(
                charts=charts, title="Test Report", output_path=report_path
            )
            assert os.path.exists(result)
            with open(result) as f:
                content = f.read()
            assert "Test Report" in content
            assert "plotly" in content.lower()

    def test_generate_report_with_metadata(self, engine, sample_ohlc_data):
        chart = engine.create_line_chart(sample_ohlc_data, "date", "close", title="Price")
        with tempfile.TemporaryDirectory() as tmpdir:
            report_path = os.path.join(tmpdir, "report.html")
            result = engine.generate_html_report(
                charts=[chart],
                title="Metadata Report",
                output_path=report_path,
                metadata={"author": "Test User", "date": "2024-01-01"},
            )
            with open(result) as f:
                content = f.read()
            assert "Test User" in content
            assert "2024-01-01" in content

    def test_generate_report_without_output_path(self, engine, sample_ohlc_data):
        chart = engine.create_line_chart(sample_ohlc_data, "date", "close", title="Price")
        result = engine.generate_html_report(charts=[chart], title="Temp Report")
        assert result is not None
        assert os.path.exists(result)
        os.unlink(result)


class TestChartExport:
    """Test chart export functionality."""

    def test_export_chart_to_html(self, engine, sample_ohlc_data):
        chart = engine.create_line_chart(sample_ohlc_data, "date", "close", title="Export Test")
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "chart.html")
            engine.export_chart(chart, output_path)
            assert os.path.exists(output_path)
            with open(output_path) as f:
                content = f.read()
            assert "Export Test" in content

    def test_export_chart_to_json(self, engine, sample_ohlc_data):
        chart = engine.create_line_chart(sample_ohlc_data, "date", "close", title="JSON Export")
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "chart.json")
            engine.export_chart(chart, output_path, format="json")
            assert os.path.exists(output_path)
            with open(output_path) as f:
                data = json.load(f)
            assert "data" in data
            assert "layout" in data


class TestFinancialMetrics:
    """Test financial metrics visualization."""

    def test_create_drawdown_chart(self, engine, sample_ohlc_data):
        chart = engine.create_drawdown_chart(
            data=sample_ohlc_data, price_col="close", title="Drawdown Analysis"
        )
        assert isinstance(chart, go.Figure)
        assert chart.data[0].type == "scatter"
        assert chart.layout.title.text == "Drawdown Analysis"

    def test_create_rolling_returns_chart(self, engine, sample_ohlc_data):
        chart = engine.create_rolling_returns_chart(
            data=sample_ohlc_data, price_col="close", window=3, title="Rolling Returns"
        )
        assert isinstance(chart, go.Figure)
        assert len(chart.data) >= 1

    def test_create_cumulative_returns_chart(self, engine, sample_returns_data):
        chart = engine.create_cumulative_returns_chart(
            data=sample_returns_data,
            columns=["AAPL", "GOOGL", "MSFT"],
            title="Cumulative Returns",
        )
        assert isinstance(chart, go.Figure)
        assert len(chart.data) == 3


class TestErrorHandling:
    """Test error handling."""

    def test_invalid_chart_type_raises_error(self, engine, sample_ohlc_data):
        with pytest.raises(ValueError, match="Unsupported chart type"):
            engine.create_dashboard(
                charts=[{"type": "invalid_type", "data": sample_ohlc_data}],
                config=DashboardConfig(),
            )

    def test_missing_column_raises_error(self, engine, sample_ohlc_data):
        with pytest.raises((KeyError, ValueError)):
            engine.create_line_chart(
                data=sample_ohlc_data, x="nonexistent_column", y="close"
            )

    def test_empty_data_raises_error(self, engine):
        empty_df = pd.DataFrame()
        with pytest.raises((ValueError, KeyError)):
            engine.create_line_chart(data=empty_df, x="date", y="close")


class TestThemeSupport:
    """Test theme and styling support."""

    def test_apply_theme(self, engine, sample_ohlc_data):
        engine.set_theme("plotly_dark")
        chart = engine.create_line_chart(sample_ohlc_data, "date", "close", title="Dark Theme")
        assert chart.layout.template.layout.plot_bgcolor == "rgb(17,17,17)"

    def test_set_custom_colors(self, engine):
        colors = ["#FF0000", "#00FF00", "#0000FF"]
        engine.set_color_palette(colors)
        assert engine._color_palette == colors

    def test_chart_uses_engine_theme(self, engine, sample_ohlc_data):
        engine.set_theme("plotly_white")
        chart = engine.create_line_chart(sample_ohlc_data, "date", "close", title="Themed")
        assert chart.layout.template.layout.plot_bgcolor == "white"


class TestChartRetrieval:
    """Test chart retrieval from engine."""

    def test_get_chart(self, engine, sample_ohlc_data):
        engine.create_line_chart(
            sample_ohlc_data, "date", "close", title="Retrievable", chart_id="ret1"
        )
        chart = engine.get_chart("ret1")
        assert chart is not None
        assert chart.layout.title.text == "Retrievable"

    def test_get_nonexistent_chart_returns_none(self, engine):
        assert engine.get_chart("nonexistent") is None

    def test_list_charts(self, engine, sample_ohlc_data):
        engine.create_line_chart(sample_ohlc_data, "date", "close", title="C1", chart_id="c1")
        engine.create_bar_chart(sample_ohlc_data, "date", "close", title="C2", chart_id="c2")
        charts = engine.list_charts()
        assert "c1" in charts
        assert "c2" in charts
        assert len(charts) == 2

    def test_clear_charts(self, engine, sample_ohlc_data):
        engine.create_line_chart(sample_ohlc_data, "date", "close", title="C1", chart_id="c1")
        engine.clear_charts()
        assert engine._charts == {}
