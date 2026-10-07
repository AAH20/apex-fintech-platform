"""Data Engineering Engine for ETL pipelines.

Provides a unified interface for data ingestion, transformation, validation,
and export using pandas DataFrames and Apache Beam transforms.

References:
    CFA Institute. (2023). Data Engineering for Investment Management.
    McKinney, W. (2022). Python for Data Analysis (3rd ed.). O'Reilly.
"""
from __future__ import annotations

import json
import os
from typing import Callable, Optional

import pandas as pd

try:
    import apache_beam as beam
    from apache_beam.options.pipeline_options import PipelineOptions
    from apache_beam.io import WriteToText
    _BEAM_AVAILABLE = True
except ImportError:
    _BEAM_AVAILABLE = False


class DataQualityError(Exception):
    """Raised when data quality validation fails."""
    pass


class DataEngineeringEngine:
    """Engine for building and executing ETL pipelines.

    Supports:
    - Data ingestion from CSV, JSON, and Python records
    - Data transformation (filter, aggregate, deduplicate, normalize, etc.)
    - Data validation and quality reporting
    - Apache Beam integration for distributed processing
    - Export to CSV, JSON, and Parquet formats
    """

    def __init__(self):
        """Initialize the Data Engineering Engine."""
        self.data: Optional[pd.DataFrame] = None

    # ------------------------------------------------------------------
    # Ingestion
    # ------------------------------------------------------------------

    def ingest_records(self, records: list[dict]) -> None:
        """Ingest data from a list of dictionaries.

        Args:
            records: List of dictionaries representing data records.
        """
        self.data = pd.DataFrame(records)

    def ingest_csv(self, filepath: str) -> None:
        """Ingest data from a CSV file.

        Args:
            filepath: Path to the CSV file.

        Raises:
            FileNotFoundError: If the file does not exist.
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"CSV file not found: {filepath}")
        self.data = pd.read_csv(filepath)

    def ingest_json(self, filepath: str) -> None:
        """Ingest data from a JSON file.

        Args:
            filepath: Path to the JSON file.

        Raises:
            FileNotFoundError: If the file does not exist.
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"JSON file not found: {filepath}")
        with open(filepath) as f:
            records = json.load(f)
        self.data = pd.DataFrame(records)

    # ------------------------------------------------------------------
    # Transformation
    # ------------------------------------------------------------------

    def add_column(self, name: str, values) -> None:
        """Add a computed column to the DataFrame.

        Args:
            name: Name of the new column.
            values: Values for the new column (Series, list, or scalar).
        """
        self.data[name] = values

    def filter_rows(self, condition: Callable) -> None:
        """Filter rows based on a condition function.

        Args:
            condition: A callable that takes a DataFrame and returns a boolean Series.
        """
        self.data = self.data[condition(self.data)].reset_index(drop=True)

    def aggregate(self, group_by: str, aggregations: dict) -> pd.DataFrame:
        """Aggregate data by a group column.

        Args:
            group_by: Column name to group by.
            aggregations: Dict mapping column names to aggregation functions.

        Returns:
            DataFrame with aggregated results.
        """
        return self.data.groupby(group_by).agg(aggregations).reset_index()

    def deduplicate(self, subset: Optional[list] = None) -> None:
        """Remove duplicate rows.

        Args:
            subset: List of columns to consider for identifying duplicates.
        """
        self.data = self.data.drop_duplicates(subset=subset).reset_index(drop=True)

    def normalize_column(self, column: str) -> None:
        """Normalize a numeric column to 0-1 range using min-max scaling.

        Args:
            column: Name of the column to normalize.
        """
        col_min = self.data[column].min()
        col_max = self.data[column].max()
        if col_max > col_min:
            self.data[column] = (self.data[column] - col_min) / (col_max - col_min)
        else:
            self.data[column] = 0.0

    def sort_by(self, column: str, ascending: bool = True) -> None:
        """Sort DataFrame by a column.

        Args:
            column: Column name to sort by.
            ascending: Sort in ascending order if True.
        """
        self.data = self.data.sort_values(by=column, ascending=ascending).reset_index(drop=True)

    def rename_columns(self, mapping: dict) -> None:
        """Rename DataFrame columns.

        Args:
            mapping: Dict mapping old column names to new column names.
        """
        self.data = self.data.rename(columns=mapping)

    def drop_columns(self, columns: list) -> None:
        """Drop specified columns.

        Args:
            columns: List of column names to drop.
        """
        self.data = self.data.drop(columns=columns)

    def fill_missing(self, fill_values: dict) -> None:
        """Fill missing values with specified values.

        Args:
            fill_values: Dict mapping column names to fill values.
        """
        self.data = self.data.fillna(fill_values)

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validate(self, rules: dict) -> dict:
        """Validate data against quality rules.

        Args:
            rules: Dict with validation rules:
                - required_columns: List of columns that must not have nulls
                - unique_columns: List of columns that must have unique values
                - ranges: Dict mapping column names to {min, max} dicts
                - custom_rules: List of {name, check} dicts where check is a callable

        Returns:
            Dict with 'valid' (bool) and 'errors' (list of error messages).
        """
        errors = []

        if self.data is None or self.data.empty:
            return {"valid": True, "errors": []}

        # Check required columns for nulls
        for col in rules.get("required_columns", []):
            if col not in self.data.columns:
                errors.append(f"Required column '{col}' not found")
            elif self.data[col].isnull().any():
                null_count = self.data[col].isnull().sum()
                errors.append(f"Column '{col}' has {null_count} null values")

        # Check unique columns
        for col in rules.get("unique_columns", []):
            if col in self.data.columns:
                dup_count = self.data[col].duplicated().sum()
                if dup_count > 0:
                    errors.append(f"Column '{col}' has {dup_count} duplicate values")

        # Check value ranges
        for col, range_spec in rules.get("ranges", {}).items():
            if col in self.data.columns:
                if "min" in range_spec:
                    below_min = (self.data[col] < range_spec["min"]).sum()
                    if below_min > 0:
                        errors.append(
                            f"Column '{col}' has {below_min} values below minimum {range_spec['min']}"
                        )
                if "max" in range_spec:
                    above_max = (self.data[col] > range_spec["max"]).sum()
                    if above_max > 0:
                        errors.append(
                            f"Column '{col}' has {above_max} values above maximum {range_spec['max']}"
                        )

        # Check custom rules
        for rule in rules.get("custom_rules", []):
            try:
                if not rule["check"](self.data):
                    errors.append(f"Custom rule '{rule['name']}' failed")
            except Exception as e:
                errors.append(f"Custom rule '{rule['name']}' raised error: {e}")

        return {"valid": len(errors) == 0, "errors": errors}

    def get_quality_report(self) -> dict:
        """Generate a data quality report.

        Returns:
            Dict with quality metrics including total_rows, total_columns,
            null_counts, and dtypes.
        """
        if self.data is None:
            return {"total_rows": 0, "total_columns": 0, "null_counts": {}, "dtypes": {}}

        return {
            "total_rows": len(self.data),
            "total_columns": len(self.data.columns),
            "null_counts": self.data.isnull().sum().to_dict(),
            "dtypes": self.data.dtypes.astype(str).to_dict(),
        }

    # ------------------------------------------------------------------
    # Pipeline Execution
    # ------------------------------------------------------------------

    def run_pipeline(
        self,
        input_path: str,
        output_path: str,
        input_format: str = "csv",
        output_format: str = "csv",
        transformations: Optional[list] = None,
        validation: Optional[dict] = None,
    ) -> None:
        """Run a full ETL pipeline.

        Args:
            input_path: Path to input data file.
            output_path: Path to output data file.
            input_format: Format of input file ('csv' or 'json').
            output_format: Format of output file ('csv', 'json', or 'parquet').
            transformations: List of transformation dicts to apply.
            validation: Validation rules dict.

        Raises:
            DataQualityError: If validation fails.
        """
        # Ingest
        if input_format == "csv":
            self.ingest_csv(input_path)
        elif input_format == "json":
            self.ingest_json(input_path)
        else:
            raise ValueError(f"Unsupported input format: {input_format}")

        # Validate
        if validation:
            result = self.validate(validation)
            if not result["valid"]:
                raise DataQualityError(f"Validation failed: {result['errors']}")

        # Transform
        if transformations:
            for transform in transformations:
                self._apply_transformation(transform)

        # Export
        if output_format == "csv":
            self.export_to_csv(output_path)
        elif output_format == "json":
            self.export_to_json(output_path)
        elif output_format == "parquet":
            self.export_to_parquet(output_path)
        else:
            raise ValueError(f"Unsupported output format: {output_format}")

    def _apply_transformation(self, transform: dict) -> None:
        """Apply a single transformation to the data.

        Args:
            transform: Dict with 'type' and transformation-specific parameters.
        """
        t_type = transform["type"]

        if t_type == "add_column":
            col_name = transform["name"]
            expr = transform["expression"]
            # Evaluate the expression in the context of the DataFrame
            self.data[col_name] = self.data.eval(expr)
        elif t_type == "filter":
            condition_str = transform["condition"]
            mask = self.data.eval(condition_str)
            self.data = self.data[mask].reset_index(drop=True)
        elif t_type == "aggregate":
            group_by = transform["group_by"]
            aggregations = transform["aggregations"]
            self.data = self.aggregate(group_by, aggregations)
        elif t_type == "deduplicate":
            subset = transform.get("subset")
            self.deduplicate(subset=subset)
        elif t_type == "normalize":
            self.normalize_column(transform["column"])
        elif t_type == "sort":
            self.sort_by(transform["column"], ascending=transform.get("ascending", True))
        elif t_type == "rename":
            self.rename_columns(transform["mapping"])
        elif t_type == "drop":
            self.drop_columns(transform["columns"])
        elif t_type == "fill_missing":
            self.fill_missing(transform["fill_values"])
        else:
            raise ValueError(f"Unknown transformation type: {t_type}")

    # ------------------------------------------------------------------
    # Apache Beam Integration
    # ------------------------------------------------------------------

    def _run_beam_and_collect(self, records: list[dict], transform_fn: Callable) -> pd.DataFrame:
        """Run a Beam pipeline and collect results into a DataFrame.

        Args:
            records: Input records as list of dicts.
            transform_fn: Function that takes a PCollection and returns a transformed PCollection.

        Returns:
            DataFrame with results.
        """
        import tempfile
        import glob

        tmp_dir = tempfile.mkdtemp()
        output_prefix = os.path.join(tmp_dir, "output")

        try:
            p = beam.Pipeline(options=PipelineOptions())
            raw = p | "Create" >> beam.Create(records)
            transformed = transform_fn(raw)
            # Serialize dicts to JSON strings before writing
            serialized = transformed | "Serialize" >> beam.Map(json.dumps)
            serialized | "Write" >> beam.io.WriteToText(output_prefix)
            p.run().wait_until_finish()

            # Read all shard files
            result_records = []
            for shard_file in sorted(glob.glob(output_prefix + "*")):
                with open(shard_file) as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            result_records.append(json.loads(line))
            return pd.DataFrame(result_records)
        finally:
            import shutil
            shutil.rmtree(tmp_dir, ignore_errors=True)

    def beam_map(self, fn: Callable) -> pd.DataFrame:
        """Apply a Beam Map transform to the data.

        Args:
            fn: Function to apply to each row (dict).

        Returns:
            DataFrame with transformed rows.
        """
        if not _BEAM_AVAILABLE:
            raise ImportError("apache_beam is required for Beam operations")

        records = self.data.to_dict("records")
        return self._run_beam_and_collect(records, lambda pc: pc | "Map" >> beam.Map(fn))

    def beam_filter(self, fn: Callable) -> pd.DataFrame:
        """Apply a Beam Filter transform to the data.

        Args:
            fn: Function that returns True for rows to keep.

        Returns:
            DataFrame with filtered rows.
        """
        if not _BEAM_AVAILABLE:
            raise ImportError("apache_beam is required for Beam operations")

        records = self.data.to_dict("records")
        return self._run_beam_and_collect(records, lambda pc: pc | "Filter" >> beam.Filter(fn))

    def beam_flatmap(self, fn: Callable) -> pd.DataFrame:
        """Apply a Beam FlatMap transform to the data.

        Args:
            fn: Function that returns a list of dicts for each input row.

        Returns:
            DataFrame with flattened rows.
        """
        if not _BEAM_AVAILABLE:
            raise ImportError("apache_beam is required for Beam operations")

        records = self.data.to_dict("records")
        return self._run_beam_and_collect(records, lambda pc: pc | "FlatMap" >> beam.FlatMap(fn))

    def run_beam_pipeline(self, input_path: str, output_path: str) -> None:
        """Run a complete Apache Beam pipeline from file to file.

        Args:
            input_path: Path to input CSV file.
            output_path: Path to output CSV file (without extension).
        """
        if not _BEAM_AVAILABLE:
            raise ImportError("apache_beam is required for Beam operations")

        import csv as csv_module
        import io as io_module

        def parse_csv_content(content):
            """Parse entire CSV content string into list of dicts."""
            reader = csv_module.reader(io_module.StringIO(content))
            rows = list(reader)
            if not rows:
                return []
            header = rows[0]
            return [dict(zip(header, row)) for row in rows[1:]]

        def format_csv_row(row):
            output = io_module.StringIO()
            writer = csv_module.DictWriter(output, fieldnames=list(row.keys()))
            writer.writerow(row)
            return output.getvalue().strip()

        # Read entire file as a single element
        with open(input_path) as f:
            file_content = f.read()

        p = beam.Pipeline(options=PipelineOptions())
        (
            p
            | "Create" >> beam.Create([file_content])
            | "ParseCSV" >> beam.FlatMap(parse_csv_content)
            | "FormatCSV" >> beam.Map(format_csv_row)
            | "Write" >> WriteToText(output_path, file_name_suffix=".csv")
        )
        p.run().wait_until_finish()

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def export_to_csv(self, filepath: str) -> None:
        """Export DataFrame to CSV.

        Args:
            filepath: Path to output CSV file.
        """
        self.data.to_csv(filepath, index=False)

    def export_to_json(self, filepath: str) -> None:
        """Export DataFrame to JSON.

        Args:
            filepath: Path to output JSON file.
        """
        self.data.to_json(filepath, orient="records", indent=2)

    def export_to_parquet(self, filepath: str) -> None:
        """Export DataFrame to Parquet.

        Args:
            filepath: Path to output Parquet file.
        """
        self.data.to_parquet(filepath, index=False)

    # ------------------------------------------------------------------
    # State Management
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Reset the engine, clearing all data."""
        self.data = None

    def get_shape(self) -> tuple:
        """Get the shape of the current DataFrame.

        Returns:
            Tuple of (rows, columns).
        """
        return self.data.shape

    def get_columns(self) -> list:
        """Get the column names of the current DataFrame.

        Returns:
            List of column names.
        """
        return list(self.data.columns)

    def head(self, n: int = 5) -> pd.DataFrame:
        """Get the first N rows of the DataFrame.

        Args:
            n: Number of rows to return.

        Returns:
            DataFrame with the first N rows.
        """
        return self.data.head(n)

    def tail(self, n: int = 5) -> pd.DataFrame:
        """Get the last N rows of the DataFrame.

        Args:
            n: Number of rows to return.

        Returns:
            DataFrame with the last N rows.
        """
        return self.data.tail(n)

    def describe(self) -> pd.DataFrame:
        """Get descriptive statistics of the DataFrame.

        Returns:
            DataFrame with descriptive statistics.
        """
        return self.data.describe()

    def info(self) -> dict:
        """Get info about the DataFrame.

        Returns:
            Dict with row count, column count, and memory usage.
        """
        return {
            "rows": len(self.data),
            "columns": len(self.data.columns),
            "memory_usage": self.data.memory_usage(deep=True).sum(),
        }
