"""Documentation Engine for Apex Fintech Platform.

Generates mkdocs-compatible documentation for all platform engines including:
- API reference documentation
- User guides with installation and usage examples
- Step-by-step tutorials
- mkdocs.yml configuration

Usage:
    from docs.engine import DocumentationEngine

    engine = DocumentationEngine()
    engine.build_all()
"""
from __future__ import annotations

import importlib
import inspect
import pkgutil
import shutil
from pathlib import Path
from types import ModuleType
from typing import Any


# Engine modules to document
ENGINE_PACKAGES = [
    "marketmaking",
    "altdata",
    "tokenization",
    "regtech",
]

# Display names for engines
ENGINE_DISPLAY_NAMES = {
    "marketmaking": "Market Making",
    "altdata": "Alternative Data",
    "tokenization": "RWA Tokenization",
    "regtech": "RegTech Compliance",
}

# Engine descriptions
ENGINE_DESCRIPTIONS = {
    "marketmaking": (
        "Avellaneda-Stoikov optimal market making engine with "
        "inventory risk management, adverse selection handling, "
        "and Hawkes process order flow modeling."
    ),
    "altdata": (
        "Alternative data aggregation and analysis engine for "
        "satellite imagery, social sentiment, and credit card "
        "transaction data with composite signal generation."
    ),
    "tokenization": (
        "ERC-3643 compliant real-world asset (RWA) tokenization "
        "framework with investor verification, transfer restrictions, "
        "and dividend distribution."
    ),
    "regtech": (
        "Regulatory compliance automation engine supporting "
        "PCI DSS, SOX, and Basel III frameworks with evidence "
        "collection and continuous monitoring."
    ),
}


class DocumentationEngine:
    """Generates comprehensive documentation for all platform engines.

    This engine discovers all engine modules, extracts their docstrings
    and API signatures, and generates mkdocs-compatible markdown files
    organized into API references, user guides, and tutorials.
    """

    def __init__(
        self,
        project_root: Path | None = None,
        output_dir: Path | None = None,
    ) -> None:
        """Initialize the documentation engine.

        Args:
            project_root: Root directory of the project. Defaults to
                the parent of the src directory.
            output_dir: Directory where documentation will be generated.
                Defaults to <project_root>/docs.
        """
        if project_root is None:
            project_root = Path(__file__).parent.parent.parent
        self.project_root = Path(project_root)

        if output_dir is None:
            output_dir = self.project_root / "docs"
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Ensure src is on the path
        src_path = str(self.project_root / "src")
        if src_path not in __import__("sys").path:
            __import__("sys").path.insert(0, src_path)

    def get_engine_names(self) -> list[str]:
        """Get list of engine package names.

        Returns:
            List of engine package names.
        """
        return list(ENGINE_PACKAGES)

    def discover_modules(self) -> list[ModuleType]:
        """Discover all engine modules in the project.

        Returns:
            List of imported module objects for all engines.
        """
        modules: list[ModuleType] = []
        for package_name in ENGINE_PACKAGES:
            try:
                package = importlib.import_module(package_name)
                modules.append(package)
                # Also discover submodules
                if hasattr(package, "__path__"):
                    for _, name, _ in pkgutil.iter_modules(package.__path__):
                        if name.startswith("_"):
                            continue
                        try:
                            full_name = f"{package_name}.{name}"
                            submodule = importlib.import_module(full_name)
                            modules.append(submodule)
                        except ImportError:
                            pass
            except ImportError:
                pass
        return modules

    def extract_docstring(self, obj: Any) -> str | None:
        """Extract the docstring from a Python object.

        Args:
            obj: A module, class, or function.

        Returns:
            The cleaned docstring, or None if not found.
        """
        doc = inspect.getdoc(obj)
        if doc is None:
            return None
        return doc.strip()

    def _get_public_classes(self, module: ModuleType) -> list[type]:
        """Get all public classes defined in a module.

        Args:
            module: The module to inspect.

        Returns:
            List of public class objects.
        """
        classes: list[type] = []
        for name, obj in inspect.getmembers(module, inspect.isclass):
            if name.startswith("_"):
                continue
            if obj.__module__ == module.__name__:
                classes.append(obj)
        return classes

    def _get_public_functions(self, module: ModuleType) -> list[Any]:
        """Get all public functions defined in a module.

        Args:
            module: The module to inspect.

        Returns:
            List of public function objects.
        """
        functions: list[Any] = []
        for name, obj in inspect.getmembers(module, inspect.isfunction):
            if name.startswith("_"):
                continue
            if obj.__module__ == module.__name__:
                functions.append(obj)
        return functions

    def _get_public_methods(self, cls: type) -> list[Any]:
        """Get all public methods of a class.

        Args:
            cls: The class to inspect.

        Returns:
            List of public method objects.
        """
        methods: list[Any] = []
        for name, obj in inspect.getmembers(cls, inspect.isfunction):
            if name.startswith("_"):
                continue
            methods.append(obj)
        return methods

    def _format_signature(self, func: Any) -> str:
        """Format a function signature for documentation.

        Args:
            func: The function to format.

        Returns:
            String representation of the signature.
        """
        try:
            sig = inspect.signature(func)
            return f"{func.__name__}{sig}"
        except (ValueError, TypeError):
            return f"{func.__name__}(...)"

    def _generate_api_for_module(self, module: ModuleType) -> str:
        """Generate API documentation markdown for a module.

        Args:
            module: The module to document.

        Returns:
            Markdown string with API documentation.
        """
        lines: list[str] = []
        module_name = module.__name__
        display_name = ENGINE_DISPLAY_NAMES.get(module_name, module_name)

        # Header
        lines.append(f"# {display_name} API Reference")
        lines.append("")

        # Module docstring
        mod_doc = self.extract_docstring(module)
        if mod_doc:
            lines.append(mod_doc)
            lines.append("")

        # Link back
        lines.append("---")
        lines.append("")
        lines.append("**Quick Links:**")
        lines.append(f"- [User Guide](../guides/{module_name}.md)")
        lines.append(f"- [Tutorial](../tutorials/{module_name}.md)")
        lines.append("")

        # Classes
        classes = self._get_public_classes(module)
        if classes:
            lines.append("## Classes")
            lines.append("")
            for cls in classes:
                lines.append(f"### `{cls.__name__}`")
                lines.append("")
                cls_doc = self.extract_docstring(cls)
                if cls_doc:
                    lines.append(cls_doc)
                    lines.append("")

                # Methods
                methods = self._get_public_methods(cls)
                if methods:
                    lines.append("#### Methods")
                    lines.append("")
                    for method in methods:
                        sig = self._format_signature(method)
                        lines.append(f"##### `{sig}`")
                        lines.append("")
                        method_doc = self.extract_docstring(method)
                        if method_doc:
                            lines.append(method_doc)
                            lines.append("")

        # Module-level functions
        functions = self._get_public_functions(module)
        if functions:
            lines.append("## Functions")
            lines.append("")
            for func in functions:
                sig = self._format_signature(func)
                lines.append(f"### `{sig}`")
                lines.append("")
                func_doc = self.extract_docstring(func)
                if func_doc:
                    lines.append(func_doc)
                    lines.append("")

        return "\n".join(lines)

    def _generate_user_guide(self, engine_name: str) -> str:
        """Generate a user guide for an engine.

        Args:
            engine_name: Name of the engine package.

        Returns:
            Markdown string with user guide content.
        """
        display_name = ENGINE_DISPLAY_NAMES.get(engine_name, engine_name)
        description = ENGINE_DESCRIPTIONS.get(engine_name, "")

        lines: list[str] = []
        lines.append(f"# {display_name} User Guide")
        lines.append("")
        lines.append("## Overview")
        lines.append("")
        lines.append(description)
        lines.append("")

        # Installation
        lines.append("## Installation")
        lines.append("")
        lines.append("Install the package with pip:")
        lines.append("")
        lines.append("```bash")
        lines.append("pip install apex-fintech-platform")
        lines.append("```")
        lines.append("")
        lines.append("Or install from source:")
        lines.append("")
        lines.append("```bash")
        lines.append("git clone https://github.com/yourorg/apex-fintech-platform.git")
        lines.append("cd apex-fintech-platform")
        lines.append("pip install -e .")
        lines.append("```")
        lines.append("")

        # Quick Start
        lines.append("## Quick Start")
        lines.append("")
        lines.append("```python")
        if engine_name == "marketmaking":
            lines.append("from marketmaking import MarketMakingEngine")
            lines.append("")
            lines.append("# Initialize the engine")
            lines.append("engine = MarketMakingEngine(")
            lines.append("    gamma=0.1,        # Risk aversion")
            lines.append("    sigma=0.5,        # Volatility")
            lines.append("    k=1.5,            # Order arrival decay")
            lines.append("    A=0.1,            # Order arrival scaling")
            lines.append("    dt=1.0,           # Time step")
            lines.append("    max_inventory=10, # Maximum inventory")
            lines.append(")")
            lines.append("")
            lines.append("# Compute optimal quotes")
            lines.append("quote = engine.compute_quotes(")
            lines.append("    mid_price=100.0,")
            lines.append("    inventory=0,")
            lines.append("    time_remaining=10.0,")
            lines.append(")")
            lines.append("print(f'Bid: {quote.bid:.2f}, Ask: {quote.ask:.2f}')")
        elif engine_name == "altdata":
            lines.append("from altdata import AlternativeDataEngine")
            lines.append("")
            lines.append("# Initialize the engine")
            lines.append("engine = AlternativeDataEngine(api_key='your_api_key')")
            lines.append("")
            lines.append("# Ingest data")
            lines.append("engine.ingest_satellite_data([")
            lines.append("    {'ticker': 'AAPL', 'parking_lot_count': 150, 'activity_score': 0.85},")
            lines.append("])")
            lines.append("")
            lines.append("# Generate composite signal")
            lines.append("signal = engine.generate_composite_signal('AAPL')")
            lines.append("print(f'Composite score: {signal[\"composite_score\"]:.2f}')")
        elif engine_name == "tokenization":
            lines.append("from tokenization import RWATokenizer")
            lines.append("")
            lines.append("# Initialize with Web3 connection")
            lines.append("tokenizer = RWATokenizer(")
            lines.append("    w3=w3,")
            lines.append("    owner_address='0x...',")
            lines.append("    token_name='Real Estate Token',")
            lines.append("    token_symbol='RET',")
            lines.append("    initial_supply=1_000_000,")
            lines.append("    asset_type='real_estate',")
            lines.append("    asset_value=500_000_000,")
            lines.append("    jurisdiction='US',")
            lines.append(")")
            lines.append("")
            lines.append("# Register and verify investor")
            lines.append("tokenizer.register_investor('0x...', 'US', accredited=True)")
            lines.append("tokenizer.verify_investor('0x...')")
        elif engine_name == "regtech":
            lines.append("from regtech import ComplianceEngine, ComplianceFramework")
            lines.append("")
            lines.append("# Initialize the engine")
            lines.append("engine = ComplianceEngine()")
            lines.append("")
            lines.append("# Register a control")
            lines.append("engine.register_control(")
            lines.append("    framework=ComplianceFramework.PCI_DSS,")
            lines.append("    control_id='PCI-1.1',")
            lines.append("    name='Firewall Configuration',")
            lines.append("    description='Maintain firewall configuration standards',")
            lines.append(")")
            lines.append("")
            lines.append("# Assess compliance")
            lines.append("engine.assess_control('PCI-1.1', ComplianceStatus.COMPLIANT)")
            lines.append("report = engine.generate_report()")
        lines.append("```")
        lines.append("")

        # Configuration
        lines.append("## Configuration")
        lines.append("")
        lines.append(f"The {display_name} engine can be configured through constructor parameters.")
        lines.append("See the [API Reference](api/{}.md) for all available options.".format(engine_name))
        lines.append("")

        # Best Practices
        lines.append("## Best Practices")
        lines.append("")
        lines.append("1. **Parameter Tuning**: Start with conservative parameters and adjust based on backtesting.")
        lines.append("2. **Error Handling**: Always wrap engine calls in try-except blocks for production use.")
        lines.append("3. **Logging**: Enable logging to track engine decisions and performance.")
        lines.append("4. **Testing**: Use the provided test suite as a reference for integration patterns.")
        lines.append("")

        # Troubleshooting
        lines.append("## Troubleshooting")
        lines.append("")
        lines.append("### Common Issues")
        lines.append("")
        lines.append("- **Import errors**: Ensure the package is installed with `pip install -e .`")
        lines.append("- **Connection errors**: Verify network connectivity and API keys")
        lines.append("- **Performance issues**: Consider batch operations where available")
        lines.append("")

        # Links
        lines.append("---")
        lines.append("")
        lines.append("**Related Documentation:**")
        lines.append(f"- [API Reference](api/{engine_name}.md)")
        lines.append(f"- [Tutorial](tutorials/{engine_name}.md)")
        lines.append("")

        return "\n".join(lines)

    def _generate_tutorial(self, engine_name: str) -> str:
        """Generate a step-by-step tutorial for an engine.

        Args:
            engine_name: Name of the engine package.

        Returns:
            Markdown string with tutorial content.
        """
        display_name = ENGINE_DISPLAY_NAMES.get(engine_name, engine_name)

        lines: list[str] = []
        lines.append(f"# {display_name} Tutorial")
        lines.append("")
        lines.append(f"This tutorial walks you through using the {display_name} engine step by step.")
        lines.append("")

        # Step 1
        lines.append("## Step 1: Setup")
        lines.append("")
        lines.append("First, ensure you have the package installed:")
        lines.append("")
        lines.append("```bash")
        lines.append("pip install apex-fintech-platform")
        lines.append("```")
        lines.append("")
        lines.append("Then import the engine:")
        lines.append("")
        lines.append("```python")
        if engine_name == "marketmaking":
            lines.append("from marketmaking import MarketMakingEngine, Quote")
            lines.append("import numpy as np")
        elif engine_name == "altdata":
            lines.append("from altdata import AlternativeDataEngine")
        elif engine_name == "tokenization":
            lines.append("from tokenization import RWATokenizer, ComplianceStatus")
        elif engine_name == "regtech":
            lines.append("from regtech import (")
            lines.append("    ComplianceEngine,")
            lines.append("    ComplianceFramework,")
            lines.append("    ComplianceStatus,")
            lines.append(")")
        lines.append("```")
        lines.append("")

        # Step 2
        lines.append("## Step 2: Initialize the Engine")
        lines.append("")
        lines.append("Create an instance of the engine with appropriate parameters:")
        lines.append("")
        lines.append("```python")
        if engine_name == "marketmaking":
            lines.append("engine = MarketMakingEngine(")
            lines.append("    gamma=0.1,        # Risk aversion coefficient")
            lines.append("    sigma=0.5,        # Asset volatility")
            lines.append("    k=1.5,            # Order arrival decay")
            lines.append("    A=0.1,            # Order arrival scaling")
            lines.append("    dt=1.0,           # Time step")
            lines.append("    max_inventory=10, # Maximum inventory limit")
            lines.append(")")
        elif engine_name == "altdata":
            lines.append("engine = AlternativeDataEngine(api_key='your_api_key')")
        elif engine_name == "tokenization":
            lines.append("# Requires a Web3 instance")
            lines.append("tokenizer = RWATokenizer(")
            lines.append("    w3=w3,")
            lines.append("    owner_address=owner_address,")
            lines.append("    token_name='My Token',")
            lines.append("    token_symbol='MTK',")
            lines.append("    initial_supply=1_000_000,")
            lines.append("    asset_type='real_estate',")
            lines.append("    asset_value=100_000_000,")
            lines.append("    jurisdiction='US',")
            lines.append(")")
        elif engine_name == "regtech":
            lines.append("engine = ComplianceEngine()")
        lines.append("```")
        lines.append("")

        # Step 3
        lines.append("## Step 3: Basic Usage")
        lines.append("")
        lines.append("Perform a basic operation:")
        lines.append("")
        lines.append("```python")
        if engine_name == "marketmaking":
            lines.append("# Compute optimal quotes")
            lines.append("quote = engine.compute_quotes(")
            lines.append("    mid_price=100.0,")
            lines.append("    inventory=0,")
            lines.append("    time_remaining=10.0,")
            lines.append(")")
            lines.append("")
            lines.append("print(f'Bid: {quote.bid:.2f}')")
            lines.append("print(f'Ask: {quote.ask:.2f}')")
            lines.append("print(f'Spread: {quote.spread:.2f}')")
            lines.append("")
            lines.append("# Expected output:")
            lines.append("# Bid: 97.50")
            lines.append("# Ask: 102.50")
            lines.append("# Spread: 5.00")
        elif engine_name == "altdata":
            lines.append("# Ingest sample data")
            lines.append("engine.ingest_sentiment_data([")
            lines.append("    {'ticker': 'AAPL', 'sentiment': 0.75, 'volume': 1000},")
            lines.append("    {'ticker': 'AAPL', 'sentiment': 0.60, 'volume': 1200},")
            lines.append("])")
            lines.append("")
            lines.append("# Analyze trends")
            lines.append("trends = engine.analyze_sentiment_trends('AAPL')")
            lines.append("print(trends)")
            lines.append("")
            lines.append("# Expected output:")
            lines.append("# {'avg_sentiment': 0.675, 'total_volume': 2200}")
        elif engine_name == "tokenization":
            lines.append("# Register an investor")
            lines.append("tokenizer.register_investor(")
            lines.append("    investor_address='0x123...',")
            lines.append("    jurisdiction='US',")
            lines.append("    accredited=True,")
            lines.append(")")
            lines.append("")
            lines.append("# Verify the investor (KYC/AML)")
            lines.append("tokenizer.verify_investor('0x123...')")
            lines.append("")
            lines.append("# Check compliance")
            lines.append("status = tokenizer.check_compliance('0x123...')")
            lines.append("print(status)  # ComplianceStatus.COMPLIANT")
        elif engine_name == "regtech":
            lines.append("# Register a compliance control")
            lines.append("engine.register_control(")
            lines.append("    framework=ComplianceFramework.PCI_DSS,")
            lines.append("    control_id='PCI-1.1',")
            lines.append("    name='Firewall Configuration',")
            lines.append("    description='Maintain firewall configuration standards',")
            lines.append(")")
            lines.append("")
            lines.append("# Add evidence")
            lines.append("engine.add_evidence(")
            lines.append("    control_id='PCI-1.1',")
            lines.append("    evidence_type='config_file',")
            lines.append("    source='firewall_rules.json',")
            lines.append("    data={'rules': 42},")
            lines.append(")")
            lines.append("")
            lines.append("# Assess the control")
            lines.append("engine.assess_control('PCI-1.1', ComplianceStatus.COMPLIANT)")
        lines.append("```")
        lines.append("")

        # Step 4
        lines.append("## Step 4: Advanced Usage")
        lines.append("")
        lines.append("Explore more advanced features:")
        lines.append("")
        lines.append("```python")
        if engine_name == "marketmaking":
            lines.append("# Batch quote computation")
            lines.append("mids = np.array([100.0, 101.0, 99.5])")
            lines.append("invs = np.array([0, 5, -3])")
            lines.append("times = np.array([10.0, 8.0, 12.0])")
            lines.append("quotes = engine.compute_quotes_batch(mids, invs, times)")
            lines.append("")
            lines.append("# Expected profit estimation")
            lines.append("profit = engine.expected_profit(quotes[0], inventory=0)")
            lines.append("print(f'Expected profit: {profit:.2f}')")
        elif engine_name == "altdata":
            lines.append("# Generate composite signal from all sources")
            lines.append("signal = engine.generate_composite_signal('AAPL')")
            lines.append("print(f'Composite score: {signal[\"composite_score\"]:.2f}')")
            lines.append("print(f'Sources used: {signal[\"data_sources_used\"]}')")
            lines.append("")
            lines.append("# Export data")
            lines.append("df = engine.export_to_dataframe('sentiment')")
            lines.append("print(df.head())")
        elif engine_name == "tokenization":
            lines.append("# Check transfer compliance")
            lines.append("can_transfer = tokenizer.can_transfer(")
            lines.append("    from_address='0x123...',")
            lines.append("    to_address='0x456...',")
            lines.append("    amount=1000,")
            lines.append(")")
            lines.append("")
            lines.append("# Distribute dividends")
            lines.append("distribution = tokenizer.distribute_dividends(total_amount=10_000)")
            lines.append("print(f'Recipients: {distribution[\"recipients\"]}')")
        elif engine_name == "regtech":
            lines.append("# Generate compliance report")
            lines.append("report = engine.generate_report()")
            lines.append("print(f'Total controls: {report.total_controls}')")
            lines.append("print(f'Compliant: {report.compliant_count}')")
            lines.append("")
            lines.append("# Check for overdue assessments")
            lines.append("overdue = engine.get_overdue_controls(max_age_days=90)")
            lines.append("print(f'Overdue controls: {len(overdue)}')")
        lines.append("```")
        lines.append("")

        # Step 5
        lines.append("## Step 5: Next Steps")
        lines.append("")
        lines.append(f"- Read the [API Reference](api/{engine_name}.md) for complete documentation")
        lines.append("- Explore the test suite for more examples")
        lines.append("- Check the [User Guide](../guides/{}.md) for best practices".format(engine_name))
        lines.append("")

        return "\n".join(lines)

    def _generate_mkdocs_yml(self) -> str:
        """Generate mkdocs.yml configuration.

        Returns:
            YAML string for mkdocs configuration.
        """
        lines: list[str] = []
        lines.append('site_name: "Apex Fintech Platform"')
        lines.append('site_description: "Documentation for the Apex Fintech Platform"')
        lines.append('site_author: "Apex Fintech"')
        lines.append("")
        lines.append('theme:')
        lines.append('  name: material')
        lines.append('  palette:')
        lines.append('    primary: indigo')
        lines.append('  features:')
        lines.append('    - navigation.sections')
        lines.append('    - navigation.top')
        lines.append('    - content.code.copy')
        lines.append("")
        lines.append('nav:')
        lines.append('  - Home: index.md')
        lines.append('  - API Reference:')
        for engine_name in ENGINE_PACKAGES:
            display = ENGINE_DISPLAY_NAMES.get(engine_name, engine_name)
            lines.append(f'    - {display}: api/{engine_name}.md')
        lines.append('  - User Guides:')
        for engine_name in ENGINE_PACKAGES:
            display = ENGINE_DISPLAY_NAMES.get(engine_name, engine_name)
            lines.append(f'    - {display}: guides/{engine_name}.md')
        lines.append('  - Tutorials:')
        for engine_name in ENGINE_PACKAGES:
            display = ENGINE_DISPLAY_NAMES.get(engine_name, engine_name)
            lines.append(f'    - {display}: tutorials/{engine_name}.md')
        lines.append("")
        lines.append('markdown_extensions:')
        lines.append('  - admonition')
        lines.append('  - pymdownx.highlight:')
        lines.append('      anchor_linenums: true')
        lines.append('  - pymdownx.superfences')
        lines.append('  - pymdownx.tabbed:')
        lines.append('      alternate_style: true')
        lines.append('  - toc:')
        lines.append('      permalink: true')
        lines.append("")
        lines.append('plugins:')
        lines.append('  - search')
        lines.append("")

        return "\n".join(lines)

    def _generate_index(self) -> str:
        """Generate the main index page.

        Returns:
            Markdown string for the index page.
        """
        lines: list[str] = []
        lines.append("# Apex Fintech Platform")
        lines.append("")
        lines.append("Welcome to the Apex Fintech Platform documentation.")
        lines.append("")
        lines.append("## Overview")
        lines.append("")
        lines.append(
            "The Apex Fintech Platform is a comprehensive suite of financial "
            "technology engines for market making, alternative data analysis, "
            "RWA tokenization, and regulatory compliance."
        )
        lines.append("")
        lines.append("## Engines")
        lines.append("")
        for engine_name in ENGINE_PACKAGES:
            display = ENGINE_DISPLAY_NAMES.get(engine_name, engine_name)
            desc = ENGINE_DESCRIPTIONS.get(engine_name, "")
            lines.append(f"### {display}")
            lines.append("")
            lines.append(desc)
            lines.append("")
            lines.append(f"- [API Reference](api/{engine_name}.md)")
            lines.append(f"- [User Guide](guides/{engine_name}.md)")
            lines.append(f"- [Tutorial](tutorials/{engine_name}.md)")
            lines.append("")
        lines.append("## Quick Start")
        lines.append("")
        lines.append("```bash")
        lines.append("pip install apex-fintech-platform")
        lines.append("```")
        lines.append("")
        lines.append("```python")
        lines.append("# Market Making")
        lines.append("from marketmaking import MarketMakingEngine")
        lines.append("engine = MarketMakingEngine(gamma=0.1, sigma=0.5, k=1.5, A=0.1, dt=1.0, max_inventory=10)")
        lines.append("")
        lines.append("# Alternative Data")
        lines.append("from altdata import AlternativeDataEngine")
        lines.append("engine = AlternativeDataEngine(api_key='your_key')")
        lines.append("")
        lines.append("# RWA Tokenization")
        lines.append("from tokenization import RWATokenizer")
        lines.append("")
        lines.append("# RegTech Compliance")
        lines.append("from regtech import ComplianceEngine")
        lines.append("engine = ComplianceEngine()")
        lines.append("```")
        lines.append("")
        lines.append("## Building Documentation")
        lines.append("")
        lines.append("To rebuild this documentation:")
        lines.append("")
        lines.append("```bash")
        lines.append("python -c \"from docs.engine import DocumentationEngine; DocumentationEngine().build_all()\"")
        lines.append("```")
        lines.append("")

        return "\n".join(lines)

    def generate_api_docs(self) -> int:
        """Generate API documentation for all engines.

        Returns:
            Number of files generated.
        """
        api_dir = self.output_dir / "api"
        api_dir.mkdir(parents=True, exist_ok=True)

        count = 0
        modules = self.discover_modules()

        # Generate docs for each discovered module
        for module in modules:
            content = self._generate_api_for_module(module)
            # Use the last part of the module name for the file
            module_file = module.__name__.replace(".", "_") + ".md"
            file_path = api_dir / module_file
            file_path.write_text(content)
            count += 1

        # Also generate combined per-engine files
        for engine_name in ENGINE_PACKAGES:
            try:
                package = importlib.import_module(engine_name)
                # Gather all submodules for this engine
                engine_modules = [package]
                if hasattr(package, "__path__"):
                    for _, name, _ in pkgutil.iter_modules(package.__path__):
                        if name.startswith("_"):
                            continue
                        try:
                            full_name = f"{engine_name}.{name}"
                            submodule = importlib.import_module(full_name)
                            engine_modules.append(submodule)
                        except ImportError:
                            pass

                # Combine all module docs into one file
                combined_lines: list[str] = []
                display_name = ENGINE_DISPLAY_NAMES.get(engine_name, engine_name)
                combined_lines.append(f"# {display_name} API Reference")
                combined_lines.append("")

                mod_doc = self.extract_docstring(package)
                if mod_doc:
                    combined_lines.append(mod_doc)
                    combined_lines.append("")

                combined_lines.append("---")
                combined_lines.append("")
                combined_lines.append("**Quick Links:**")
                combined_lines.append(f"- [User Guide](../guides/{engine_name}.md)")
                combined_lines.append(f"- [Tutorial](../tutorials/{engine_name}.md)")
                combined_lines.append("")

                for mod in engine_modules:
                    # Include submodule docstrings (e.g., regtech.compliance)
                    if mod is not package:
                        sub_doc = self.extract_docstring(mod)
                        if sub_doc:
                            combined_lines.append(sub_doc)
                            combined_lines.append("")

                    classes = self._get_public_classes(mod)
                    functions = self._get_public_functions(mod)

                    if classes:
                        combined_lines.append("## Classes")
                        combined_lines.append("")
                        for cls in classes:
                            combined_lines.append(f"### `{cls.__name__}`")
                            combined_lines.append("")
                            cls_doc = self.extract_docstring(cls)
                            if cls_doc:
                                combined_lines.append(cls_doc)
                                combined_lines.append("")

                            methods = self._get_public_methods(cls)
                            if methods:
                                combined_lines.append("#### Methods")
                                combined_lines.append("")
                                for method in methods:
                                    sig = self._format_signature(method)
                                    combined_lines.append(f"##### `{sig}`")
                                    combined_lines.append("")
                                    method_doc = self.extract_docstring(method)
                                    if method_doc:
                                        combined_lines.append(method_doc)
                                        combined_lines.append("")

                    if functions:
                        combined_lines.append("## Functions")
                        combined_lines.append("")
                        for func in functions:
                            sig = self._format_signature(func)
                            combined_lines.append(f"### `{sig}`")
                            combined_lines.append("")
                            func_doc = self.extract_docstring(func)
                            if func_doc:
                                combined_lines.append(func_doc)
                                combined_lines.append("")

                file_path = api_dir / f"{engine_name}.md"
                file_path.write_text("\n".join(combined_lines))
                count += 1
            except ImportError:
                # Generate a placeholder for missing modules
                display = ENGINE_DISPLAY_NAMES.get(engine_name, engine_name)
                content = f"# {display} API Reference\n\nDocumentation for {display} engine.\n"
                file_path = api_dir / f"{engine_name}.md"
                file_path.write_text(content)
                count += 1

        return count

    def generate_user_guides(self) -> int:
        """Generate user guides for all engines.

        Returns:
            Number of files generated.
        """
        guide_dir = self.output_dir / "guides"
        guide_dir.mkdir(parents=True, exist_ok=True)

        count = 0
        for engine_name in ENGINE_PACKAGES:
            content = self._generate_user_guide(engine_name)
            file_path = guide_dir / f"{engine_name}.md"
            file_path.write_text(content)
            count += 1

        return count

    def generate_tutorials(self) -> int:
        """Generate tutorials for all engines.

        Returns:
            Number of files generated.
        """
        tutorial_dir = self.output_dir / "tutorials"
        tutorial_dir.mkdir(parents=True, exist_ok=True)

        count = 0
        for engine_name in ENGINE_PACKAGES:
            content = self._generate_tutorial(engine_name)
            file_path = tutorial_dir / f"{engine_name}.md"
            file_path.write_text(content)
            count += 1

        return count

    def generate_mkdocs_yml(self) -> int:
        """Generate mkdocs.yml configuration.

        Returns:
            Number of files generated (1).
        """
        self.output_dir.mkdir(parents=True, exist_ok=True)
        content = self._generate_mkdocs_yml()
        file_path = self.output_dir / "mkdocs.yml"
        file_path.write_text(content)
        return 1

    def generate_index(self) -> int:
        """Generate the main index page.

        Returns:
            Number of files generated (1).
        """
        self.output_dir.mkdir(parents=True, exist_ok=True)
        content = self._generate_index()
        file_path = self.output_dir / "index.md"
        file_path.write_text(content)
        return 1

    def build_all(self) -> int:
        """Generate all documentation.

        This is the main entry point for generating the complete
        documentation site. It creates:
        - API reference docs in docs/api/
        - User guides in docs/guides/
        - Tutorials in docs/tutorials/
        - mkdocs.yml configuration
        - index.md main page

        Returns:
            Total number of files generated.
        """
        # Clean output directory
        if self.output_dir.exists():
            shutil.rmtree(self.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        count = 0
        count += self.generate_api_docs()
        count += self.generate_user_guides()
        count += self.generate_tutorials()
        count += self.generate_mkdocs_yml()
        count += self.generate_index()

        return count
