"""Tests for the Data Engineering Engine.

Covers data ingestion, transformation, validation, and pipeline execution
using pandas DataFrames and Apache Beam transforms.
"""
import json
import os

import pandas as pd
import pytest

from src.data.engineering import DataEngineeringEngine, DataQualityError


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def engine():
    """Create a fresh DataEngineeringEngine instance."""
    return DataEngineeringEngine()


@pytest.fixture
def sample_trades():
    """Sample trade records for testing."""
    return [
        {"trade_id": "T001", "symbol": "AAPL", "price": 150.0, "quantity": 100, "side": "buy"},
        {"trade_id": "T002", "symbol": "AAPL", "price": 151.0, "quantity": 200, "side": "sell"},
        {"trade_id": "T003", "symbol": "GOOGL", "price": 2800.0, "quantity": 50, "side": "buy"},
        {"trade_id": "T004", "symbol": "GOOGL", "price": 2805.0, "quantity": 75, "side": "sell"},
        {"trade_id": "T005", "symbol": "AAPL", "price": 149.5, "quantity": 150, "side": "buy"},
    ]


@pytest.fixture
def sample_csv(tmp_path):
    """Create a temporary CSV file with sample data."""
    csv_path = tmp_path / "trades.csv"
    df = pd.DataFrame([
        {"trade_id": "T001", "symbol": "AAPL", "price": 150.0, "quantity": 100, "side": "buy"},
        {"trade_id": "T002", "symbol": "AAPL", "price": 151.0, "quantity": 200, "side": "sell"},
        {"trade_id": "T003", "symbol": "GOOGL", "price": 2800.0, "quantity": 50, "side": "buy"},
    ])
    df.to_csv(csv_path, index=False)
    return str(csv_path)


@pytest.fixture
def sample_json(tmp_path):
    """Create a temporary JSON file with sample data."""
    json_path = tmp_path / "trades.json"
    data = [
        {"trade_id": "T001", "symbol": "AAPL", "price": 150.0, "quantity": 100, "side": "buy"},
        {"trade_id": "T002", "symbol": "AAPL", "price": 151.0, "quantity": 200, "side": "sell"},
    ]
    with open(json_path, "w") as f:
        json.dump(data, f)
    return str(json_path)


@pytest.fixture
def sample_df(sample_trades):
    """Create a DataFrame from sample trades."""
    return pd.DataFrame(sample_trades)


# ---------------------------------------------------------------------------
# Ingestion Tests
# ---------------------------------------------------------------------------


class TestDataIngestion:
    """Test data ingestion from various sources."""

    def test_ingest_from_records(self, engine, sample_trades):
        """Ingest data from a list of dictionaries."""
        engine.ingest_records(sample_trades)
        assert engine.data is not None
        assert len(engine.data) == 5
        assert list(engine.data.columns) == ["trade_id", "symbol", "price", "quantity", "side"]

    def test_ingest_from_csv(self, engine, sample_csv):
        """Ingest data from a CSV file."""
        engine.ingest_csv(sample_csv)
        assert engine.data is not None
        assert len(engine.data) == 3
        assert "symbol" in engine.data.columns

    def test_ingest_from_json(self, engine, sample_json):
        """Ingest data from a JSON file."""
        engine.ingest_json(sample_json)
        assert engine.data is not None
        assert len(engine.data) == 2
        assert "price" in engine.data.columns

    def test_ingest_csv_missing_file_raises(self, engine):
        """Ingesting a non-existent CSV raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            engine.ingest_csv("/nonexistent/path/data.csv")

    def test_ingest_json_missing_file_raises(self, engine):
        """Ingesting a non-existent JSON raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            engine.ingest_json("/nonexistent/path/data.json")

    def test_ingest_records_empty_list(self, engine):
        """Ingesting an empty list creates an empty DataFrame."""
        engine.ingest_records([])
        assert engine.data is not None
        assert len(engine.data) == 0

    def test_ingest_records_preserves_dtypes(self, engine, sample_trades):
        """Ingested records preserve correct data types."""
        engine.ingest_records(sample_trades)
        assert engine.data["price"].dtype == float
        assert engine.data["quantity"].dtype == int


# ---------------------------------------------------------------------------
# Transformation Tests
# ---------------------------------------------------------------------------


class TestDataTransformation:
    """Test data transformation operations."""

    def test_add_computed_column(self, engine, sample_df):
        """Add a computed column (notional = price * quantity)."""
        engine.data = sample_df.copy()
        engine.add_column("notional", engine.data["price"] * engine.data["quantity"])
        assert "notional" in engine.data.columns
        assert engine.data["notional"].iloc[0] == 15000.0

    def test_filter_rows(self, engine, sample_df):
        """Filter rows based on a condition."""
        engine.data = sample_df.copy()
        engine.filter_rows(lambda df: df["symbol"] == "AAPL")
        assert len(engine.data) == 3
        assert all(engine.data["symbol"] == "AAPL")

    def test_aggregate_by_group(self, engine, sample_df):
        """Aggregate data by a group column."""
        engine.data = sample_df.copy()
        result = engine.aggregate(group_by="symbol", aggregations={"price": "mean", "quantity": "sum"})
        assert len(result) == 2
        aapl_row = result[result["symbol"] == "AAPL"].iloc[0]
        assert aapl_row["price"] == pytest.approx(150.1667, rel=0.01)
        assert aapl_row["quantity"] == 450

    def test_deduplicate(self, engine):
        """Remove duplicate rows based on key columns."""
        records = [
            {"trade_id": "T001", "symbol": "AAPL", "price": 150.0},
            {"trade_id": "T001", "symbol": "AAPL", "price": 150.0},
            {"trade_id": "T002", "symbol": "GOOGL", "price": 2800.0},
        ]
        engine.ingest_records(records)
        engine.deduplicate(subset=["trade_id"])
        assert len(engine.data) == 2

    def test_normalize_column(self, engine, sample_df):
        """Normalize a numeric column to 0-1 range."""
        engine.data = sample_df.copy()
        engine.normalize_column("price")
        assert engine.data["price"].min() == pytest.approx(0.0, abs=0.01)
        assert engine.data["price"].max() == pytest.approx(1.0, abs=0.01)

    def test_sort_by_column(self, engine, sample_df):
        """Sort DataFrame by a column."""
        engine.data = sample_df.copy()
        engine.sort_by("price", ascending=False)
        assert engine.data["price"].iloc[0] == 2805.0
        assert engine.data["price"].iloc[-1] == 149.5

    def test_rename_columns(self, engine, sample_df):
        """Rename DataFrame columns."""
        engine.data = sample_df.copy()
        engine.rename_columns({"price": "trade_price", "quantity": "trade_qty"})
        assert "trade_price" in engine.data.columns
        assert "trade_qty" in engine.data.columns
        assert "price" not in engine.data.columns

    def test_drop_columns(self, engine, sample_df):
        """Drop specified columns."""
        engine.data = sample_df.copy()
        engine.drop_columns(["side", "quantity"])
        assert "side" not in engine.data.columns
        assert "quantity" not in engine.data.columns
        assert "trade_id" in engine.data.columns

    def test_fill_missing_values(self, engine):
        """Fill missing values with a specified value."""
        records = [
            {"trade_id": "T001", "symbol": "AAPL", "price": 150.0},
            {"trade_id": "T002", "symbol": None, "price": 151.0},
            {"trade_id": "T003", "symbol": "GOOGL", "price": None},
        ]
        engine.ingest_records(records)
        engine.fill_missing({"symbol": "UNKNOWN", "price": 0.0})
        assert engine.data["symbol"].iloc[1] == "UNKNOWN"
        assert engine.data["price"].iloc[2] == 0.0


# ---------------------------------------------------------------------------
# Validation Tests
# ---------------------------------------------------------------------------


class TestDataValidation:
    """Test data validation and quality checks."""

    def test_validate_no_nulls_passes(self, engine, sample_df):
        """Validation passes when no nulls exist in required columns."""
        engine.data = sample_df.copy()
        result = engine.validate({"required_columns": ["trade_id", "symbol", "price"]})
        assert result["valid"] is True
        assert result["errors"] == []

    def test_validate_no_nulls_fails(self, engine):
        """Validation fails when nulls exist in required columns."""
        records = [
            {"trade_id": "T001", "symbol": "AAPL", "price": 150.0},
            {"trade_id": None, "symbol": "GOOGL", "price": 2800.0},
        ]
        engine.ingest_records(records)
        result = engine.validate({"required_columns": ["trade_id", "symbol", "price"]})
        assert result["valid"] is False
        assert len(result["errors"]) > 0

    def test_validate_unique_values_passes(self, engine, sample_df):
        """Validation passes when column values are unique."""
        engine.data = sample_df.copy()
        result = engine.validate({"unique_columns": ["trade_id"]})
        assert result["valid"] is True

    def test_validate_unique_values_fails(self, engine):
        """Validation fails when column values are not unique."""
        records = [
            {"trade_id": "T001", "symbol": "AAPL", "price": 150.0},
            {"trade_id": "T001", "symbol": "GOOGL", "price": 2800.0},
        ]
        engine.ingest_records(records)
        result = engine.validate({"unique_columns": ["trade_id"]})
        assert result["valid"] is False

    def test_validate_value_range_passes(self, engine, sample_df):
        """Validation passes when values are within range."""
        engine.data = sample_df.copy()
        result = engine.validate({"ranges": {"price": {"min": 100.0, "max": 3000.0}}})
        assert result["valid"] is True

    def test_validate_value_range_fails(self, engine, sample_df):
        """Validation fails when values are outside range."""
        engine.data = sample_df.copy()
        result = engine.validate({"ranges": {"price": {"min": 200.0, "max": 3000.0}}})
        assert result["valid"] is False

    def test_validate_custom_rule(self, engine, sample_df):
        """Validation with a custom rule function."""
        engine.data = sample_df.copy()
        result = engine.validate({
            "custom_rules": [
                {"name": "all_positive_quantity", "check": lambda df: (df["quantity"] > 0).all()}
            ]
        })
        assert result["valid"] is True

    def test_validate_custom_rule_fails(self, engine):
        """Validation fails when custom rule returns False."""
        records = [
            {"trade_id": "T001", "symbol": "AAPL", "price": 150.0, "quantity": 100},
            {"trade_id": "T002", "symbol": "GOOGL", "price": 2800.0, "quantity": -50},
        ]
        engine.ingest_records(records)
        result = engine.validate({
            "custom_rules": [
                {"name": "all_positive_quantity", "check": lambda df: (df["quantity"] > 0).all()}
            ]
        })
        assert result["valid"] is False

    def test_get_quality_report(self, engine, sample_df):
        """Generate a data quality report."""
        engine.data = sample_df.copy()
        report = engine.get_quality_report()
        assert report["total_rows"] == 5
        assert report["total_columns"] == 5
        assert "null_counts" in report
        assert "dtypes" in report

    def test_validate_empty_dataframe(self, engine):
        """Validation on empty DataFrame returns valid with warning."""
        engine.ingest_records([])
        result = engine.validate({"required_columns": ["trade_id"]})
        assert result["valid"] is True


# ---------------------------------------------------------------------------
# Pipeline Tests
# ---------------------------------------------------------------------------


class TestPipelineExecution:
    """Test full ETL pipeline execution."""

    def test_run_pipeline_csv_to_csv(self, engine, sample_csv, tmp_path):
        """Run a full ETL pipeline from CSV to CSV."""
        output_path = str(tmp_path / "output.csv")
        engine.run_pipeline(
            input_path=sample_csv,
            output_path=output_path,
            input_format="csv",
            output_format="csv",
            transformations=[
                {"type": "add_column", "name": "notional", "expression": "price * quantity"},
            ],
            validation={"required_columns": ["trade_id", "symbol"]},
        )
        assert os.path.exists(output_path)
        result = pd.read_csv(output_path)
        assert "notional" in result.columns
        assert len(result) == 3

    def test_run_pipeline_with_filter(self, engine, sample_csv, tmp_path):
        """Run pipeline with a filter transformation."""
        output_path = str(tmp_path / "filtered.csv")
        engine.run_pipeline(
            input_path=sample_csv,
            output_path=output_path,
            input_format="csv",
            output_format="csv",
            transformations=[
                {"type": "filter", "condition": "symbol == 'AAPL'"},
            ],
        )
        result = pd.read_csv(output_path)
        assert len(result) == 2
        assert all(result["symbol"] == "AAPL")

    def test_run_pipeline_with_aggregation(self, engine, sample_csv, tmp_path):
        """Run pipeline with aggregation transformation."""
        output_path = str(tmp_path / "aggregated.csv")
        engine.run_pipeline(
            input_path=sample_csv,
            output_path=output_path,
            input_format="csv",
            output_format="csv",
            transformations=[
                {"type": "aggregate", "group_by": "symbol", "aggregations": {"price": "mean", "quantity": "sum"}},
            ],
        )
        result = pd.read_csv(output_path)
        assert len(result) == 2
        assert "symbol" in result.columns

    def test_run_pipeline_validation_failure_raises(self, engine, sample_csv, tmp_path):
        """Pipeline raises DataQualityError when validation fails."""
        output_path = str(tmp_path / "output.csv")
        with pytest.raises(DataQualityError):
            engine.run_pipeline(
                input_path=sample_csv,
                output_path=output_path,
                input_format="csv",
                output_format="csv",
                validation={"required_columns": ["nonexistent_column"]},
            )

    def test_run_pipeline_json_to_csv(self, engine, sample_json, tmp_path):
        """Run pipeline from JSON input to CSV output."""
        output_path = str(tmp_path / "output.csv")
        engine.run_pipeline(
            input_path=sample_json,
            output_path=output_path,
            input_format="json",
            output_format="csv",
        )
        assert os.path.exists(output_path)
        result = pd.read_csv(output_path)
        assert len(result) == 2

    def test_run_pipeline_with_dedup(self, engine, tmp_path):
        """Run pipeline with deduplication."""
        csv_path = tmp_path / "dupes.csv"
        pd.DataFrame([
            {"trade_id": "T001", "symbol": "AAPL", "price": 150.0},
            {"trade_id": "T001", "symbol": "AAPL", "price": 150.0},
            {"trade_id": "T002", "symbol": "GOOGL", "price": 2800.0},
        ]).to_csv(csv_path, index=False)

        output_path = str(tmp_path / "deduped.csv")
        engine.run_pipeline(
            input_path=str(csv_path),
            output_path=output_path,
            input_format="csv",
            output_format="csv",
            transformations=[
                {"type": "deduplicate", "subset": ["trade_id"]},
            ],
        )
        result = pd.read_csv(output_path)
        assert len(result) == 2


# ---------------------------------------------------------------------------
# Export Tests
# ---------------------------------------------------------------------------


class TestDataExport:
    """Test data export functionality."""

    def test_export_to_csv(self, engine, sample_df, tmp_path):
        """Export DataFrame to CSV."""
        engine.data = sample_df.copy()
        output_path = str(tmp_path / "export.csv")
        engine.export_to_csv(output_path)
        assert os.path.exists(output_path)
        result = pd.read_csv(output_path)
        assert len(result) == 5

    def test_export_to_json(self, engine, sample_df, tmp_path):
        """Export DataFrame to JSON."""
        engine.data = sample_df.copy()
        output_path = str(tmp_path / "export.json")
        engine.export_to_json(output_path)
        assert os.path.exists(output_path)
        with open(output_path) as f:
            data = json.load(f)
        assert len(data) == 5

    def test_export_to_parquet(self, engine, sample_df, tmp_path):
        """Export DataFrame to Parquet."""
        engine.data = sample_df.copy()
        output_path = str(tmp_path / "export.parquet")
        engine.export_to_parquet(output_path)
        assert os.path.exists(output_path)
        result = pd.read_parquet(output_path)
        assert len(result) == 5


# ---------------------------------------------------------------------------
# Apache Beam Integration Tests
# ---------------------------------------------------------------------------


class TestBeamIntegration:
    """Test Apache Beam pipeline integration."""

    def test_beam_transform_map(self, engine, sample_df):
        """Apply a Beam Map transform to the data."""
        engine.data = sample_df.copy()
        result = engine.beam_map(lambda row: {**row, "notional": row["price"] * row["quantity"]})
        assert "notional" in result.columns
        assert result["notional"].iloc[0] == 15000.0

    def test_beam_transform_filter(self, engine, sample_df):
        """Apply a Beam Filter transform to the data."""
        engine.data = sample_df.copy()
        result = engine.beam_filter(lambda row: row["symbol"] == "AAPL")
        assert len(result) == 3
        assert all(result["symbol"] == "AAPL")

    def test_beam_transform_flatmap(self, engine, sample_df):
        """Apply a Beam FlatMap transform to the data."""
        engine.data = sample_df.copy()
        # FlatMap: split each row into two rows (buy/sell indicator)
        def split_row(row):
            return [
                {**row, "order_type": "market"},
                {**row, "order_type": "limit"},
            ]
        result = engine.beam_flatmap(split_row)
        assert len(result) == len(sample_df) * 2
        assert "order_type" in result.columns

    def test_beam_pipeline_end_to_end(self, engine, sample_csv, tmp_path):
        """Run a complete Beam pipeline from file to file."""
        output_prefix = str(tmp_path / "beam_output")
        engine.run_beam_pipeline(
            input_path=sample_csv,
            output_path=output_prefix,
        )
        # WriteToText creates sharded files
        import glob
        output_files = glob.glob(output_prefix + "*")
        assert len(output_files) > 0
        result = pd.read_csv(output_files[0], header=None)
        assert len(result) == 3


# ---------------------------------------------------------------------------
# Engine State Tests
# ---------------------------------------------------------------------------


class TestEngineState:
    """Test engine state management."""

    def test_initial_state(self, engine):
        """Engine starts with no data."""
        assert engine.data is None

    def test_reset_clears_data(self, engine, sample_df):
        """Reset clears all ingested data."""
        engine.data = sample_df.copy()
        engine.reset()
        assert engine.data is None

    def test_get_shape(self, engine, sample_df):
        """Get the shape of the current DataFrame."""
        engine.data = sample_df.copy()
        assert engine.get_shape() == (5, 5)

    def test_get_columns(self, engine, sample_df):
        """Get the column names of the current DataFrame."""
        engine.data = sample_df.copy()
        cols = engine.get_columns()
        assert "trade_id" in cols
        assert "symbol" in cols
        assert "price" in cols

    def test_head(self, engine, sample_df):
        """Get the first N rows of the DataFrame."""
        engine.data = sample_df.copy()
        result = engine.head(2)
        assert len(result) == 2
        assert result.iloc[0]["trade_id"] == "T001"

    def test_tail(self, engine, sample_df):
        """Get the last N rows of the DataFrame."""
        engine.data = sample_df.copy()
        result = engine.tail(2)
        assert len(result) == 2
        assert result.iloc[-1]["trade_id"] == "T005"

    def test_describe(self, engine, sample_df):
        """Get descriptive statistics of the DataFrame."""
        engine.data = sample_df.copy()
        desc = engine.describe()
        assert "price" in desc.columns
        assert "quantity" in desc.columns
        assert desc.loc["mean", "price"] == pytest.approx(1211.1, rel=0.01)

    def test_info_returns_dict(self, engine, sample_df):
        """Get info about the DataFrame as a dictionary."""
        engine.data = sample_df.copy()
        info = engine.info()
        assert isinstance(info, dict)
        assert info["rows"] == 5
        assert info["columns"] == 5
