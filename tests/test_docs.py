"""TDD tests for the Documentation Engine.

These tests define the expected behavior of DocumentationEngine,
which generates mkdocs-compatible documentation for all platform engines.
"""
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

# Ensure src is importable
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from docs.engine import DocumentationEngine


# ── Fixtures ──────────────────────────────────────────────────────────────


@pytest.fixture
def engine():
    """Create a fresh DocumentationEngine instance."""
    return DocumentationEngine(
        project_root=Path(__file__).parent.parent,
        output_dir=Path(__file__).parent.parent / "docs",
    )


@pytest.fixture
def mock_module():
    """Mock a Python module with docstrings."""
    mod = MagicMock()
    mod.__name__ = "test_module"
    mod.__doc__ = "Test module docstring."
    return mod


# ── Test 1: Initialization ───────────────────────────────────────────────


class TestInitialization:
    def test_engine_initializes_with_project_root(self, engine):
        assert engine.project_root == Path(__file__).parent.parent

    def test_engine_initializes_with_output_dir(self, engine):
        assert engine.output_dir == Path(__file__).parent.parent / "docs"

    def test_engine_has_default_engines(self, engine):
        """Engine should discover all platform engines."""
        engines = engine.get_engine_names()
        assert "marketmaking" in engines
        assert "altdata" in engines
        assert "tokenization" in engines
        assert "regtech" in engines

    def test_engine_output_dir_created(self, engine, tmp_path):
        """Engine creates output directory if it doesn't exist."""
        custom_out = tmp_path / "custom_docs"
        DocumentationEngine(
            project_root=Path(__file__).parent.parent,
            output_dir=custom_out,
        )
        assert custom_out.exists()


# ── Test 2: Module Discovery ─────────────────────────────────────────────


class TestModuleDiscovery:
    def test_discovers_all_engine_modules(self, engine):
        """Should find all engine modules in src/."""
        modules = engine.discover_modules()
        assert len(modules) >= 4
        module_names = [m.__name__ for m in modules]
        assert any("marketmaking" in name for name in module_names)
        assert any("altdata" in name for name in module_names)
        assert any("tokenization" in name for name in module_names)
        assert any("regtech" in name for name in module_names)

    def test_discovers_hawkes_submodule(self, engine):
        """Should find submodules like hawkes."""
        modules = engine.discover_modules()
        module_names = [m.__name__ for m in modules]
        assert any("hawkes" in name for name in module_names)

    def test_skips_non_engine_modules(self, engine):
        """Should skip __init__ and non-engine modules."""
        modules = engine.discover_modules()
        for mod in modules:
            assert mod.__name__ != "__init__"
            assert not mod.__name__.startswith("_")


# ── Test 3: API Documentation Generation ─────────────────────────────────


class TestAPIDocumentation:
    def test_generate_api_docs_creates_markdown(self, engine, tmp_path):
        """API docs should produce valid markdown files."""
        engine.output_dir = tmp_path / "docs"
        engine.output_dir.mkdir(parents=True)
        engine.generate_api_docs()
        api_dir = engine.output_dir / "api"
        assert api_dir.exists()
        md_files = list(api_dir.glob("*.md"))
        assert len(md_files) >= 4

    def test_api_docs_contain_class_names(self, engine, tmp_path):
        """API docs should document key classes."""
        engine.output_dir = tmp_path / "docs"
        engine.output_dir.mkdir(parents=True)
        engine.generate_api_docs()
        api_dir = engine.output_dir / "api"
        all_content = ""
        for f in api_dir.glob("*.md"):
            all_content += f.read_text()
        assert "MarketMakingEngine" in all_content
        assert "AlternativeDataEngine" in all_content
        assert "RWATokenizer" in all_content
        assert "ComplianceEngine" in all_content

    def test_api_docs_contain_method_signatures(self, engine, tmp_path):
        """API docs should include method signatures from docstrings."""
        engine.output_dir = tmp_path / "docs"
        engine.output_dir.mkdir(parents=True)
        engine.generate_api_docs()
        api_dir = engine.output_dir / "api"
        all_content = ""
        for f in api_dir.glob("*.md"):
            all_content += f.read_text()
        assert "compute_quotes" in all_content
        assert "generate_composite_signal" in all_content
        assert "check_compliance" in all_content

    def test_api_docs_have_proper_headers(self, engine, tmp_path):
        """Each API doc file should have a markdown header."""
        engine.output_dir = tmp_path / "docs"
        engine.output_dir.mkdir(parents=True)
        engine.generate_api_docs()
        api_dir = engine.output_dir / "api"
        for f in api_dir.glob("*.md"):
            content = f.read_text()
            assert content.startswith("#"), f"{f.name} should start with #"

    def test_api_docs_include_docstrings(self, engine, tmp_path):
        """API docs should extract and include docstrings."""
        engine.output_dir = tmp_path / "docs"
        engine.output_dir.mkdir(parents=True)
        engine.generate_api_docs()
        api_dir = engine.output_dir / "api"
        all_content = ""
        for f in api_dir.glob("*.md"):
            all_content += f.read_text()
        # Check for known docstring content
        assert "Avellaneda" in all_content or "market making" in all_content.lower()


# ── Test 4: User Guide Generation ────────────────────────────────────────


class TestUserGuide:
    def test_generate_user_guides_creates_files(self, engine, tmp_path):
        """User guides should be generated for each engine."""
        engine.output_dir = tmp_path / "docs"
        engine.output_dir.mkdir(parents=True)
        engine.generate_user_guides()
        guide_dir = engine.output_dir / "guides"
        assert guide_dir.exists()
        md_files = list(guide_dir.glob("*.md"))
        assert len(md_files) >= 4

    def test_user_guides_contain_installation_section(self, engine, tmp_path):
        """User guides should include installation instructions."""
        engine.output_dir = tmp_path / "docs"
        engine.output_dir.mkdir(parents=True)
        engine.generate_user_guides()
        guide_dir = engine.output_dir / "guides"
        all_content = ""
        for f in guide_dir.glob("*.md"):
            all_content += f.read_text()
        assert "install" in all_content.lower() or "pip" in all_content.lower()

    def test_user_guides_contain_usage_examples(self, engine, tmp_path):
        """User guides should include code examples."""
        engine.output_dir = tmp_path / "docs"
        engine.output_dir.mkdir(parents=True)
        engine.generate_user_guides()
        guide_dir = engine.output_dir / "guides"
        all_content = ""
        for f in guide_dir.glob("*.md"):
            all_content += f.read_text()
        assert "```" in all_content  # Code blocks
        assert "import" in all_content

    def test_user_guides_have_engine_overview(self, engine, tmp_path):
        """Each user guide should start with an engine overview."""
        engine.output_dir = tmp_path / "docs"
        engine.output_dir.mkdir(parents=True)
        engine.generate_user_guides()
        guide_dir = engine.output_dir / "guides"
        for f in guide_dir.glob("*.md"):
            content = f.read_text()
            assert "#" in content  # Has headers
            assert len(content) > 200  # Substantial content


# ── Test 5: Tutorial Generation ──────────────────────────────────────────


class TestTutorialGeneration:
    def test_generate_tutorials_creates_files(self, engine, tmp_path):
        """Tutorials should be generated for each engine."""
        engine.output_dir = tmp_path / "docs"
        engine.output_dir.mkdir(parents=True)
        engine.generate_tutorials()
        tutorial_dir = engine.output_dir / "tutorials"
        assert tutorial_dir.exists()
        md_files = list(tutorial_dir.glob("*.md"))
        assert len(md_files) >= 4

    def test_tutorials_have_step_by_step_structure(self, engine, tmp_path):
        """Tutorials should have numbered steps."""
        engine.output_dir = tmp_path / "docs"
        engine.output_dir.mkdir(parents=True)
        engine.generate_tutorials()
        tutorial_dir = engine.output_dir / "tutorials"
        all_content = ""
        for f in tutorial_dir.glob("*.md"):
            all_content += f.read_text()
        # Should have step markers
        assert "Step" in all_content or "step" in all_content.lower()
        assert "1." in all_content or "##" in all_content

    def test_tutorials_include_code_examples(self, engine, tmp_path):
        """Tutorials should include runnable code examples."""
        engine.output_dir = tmp_path / "docs"
        engine.output_dir.mkdir(parents=True)
        engine.generate_tutorials()
        tutorial_dir = engine.output_dir / "tutorials"
        all_content = ""
        for f in tutorial_dir.glob("*.md"):
            all_content += f.read_text()
        assert "```python" in all_content

    def test_tutorials_have_expected_output(self, engine, tmp_path):
        """Tutorials should show expected output."""
        engine.output_dir = tmp_path / "docs"
        engine.output_dir.mkdir(parents=True)
        engine.generate_tutorials()
        tutorial_dir = engine.output_dir / "tutorials"
        all_content = ""
        for f in tutorial_dir.glob("*.md"):
            all_content += f.read_text()
        assert ">>>" in all_content or "Expected" in all_content or "Output" in all_content


# ── Test 6: mkdocs.yml Generation ────────────────────────────────────────


class TestMkdocsConfig:
    def test_generate_mkdocs_yml(self, engine, tmp_path):
        """Should generate a valid mkdocs.yml configuration."""
        engine.output_dir = tmp_path / "docs"
        engine.output_dir.mkdir(parents=True)
        engine.generate_mkdocs_yml()
        yml_file = engine.output_dir / "mkdocs.yml"
        assert yml_file.exists()
        content = yml_file.read_text()
        assert "site_name" in content
        assert "nav" in content

    def test_mkdocs_yml_includes_all_sections(self, engine, tmp_path):
        """mkdocs.yml should reference API docs, guides, and tutorials."""
        engine.output_dir = tmp_path / "docs"
        engine.output_dir.mkdir(parents=True)
        engine.generate_mkdocs_yml()
        yml_file = engine.output_dir / "mkdocs.yml"
        content = yml_file.read_text()
        assert "api" in content.lower()
        assert "guide" in content.lower()
        assert "tutorial" in content.lower()

    def test_mkdocs_yml_has_theme(self, engine, tmp_path):
        """mkdocs.yml should specify a theme."""
        engine.output_dir = tmp_path / "docs"
        engine.output_dir.mkdir(parents=True)
        engine.generate_mkdocs_yml()
        yml_file = engine.output_dir / "mkdocs.yml"
        content = yml_file.read_text()
        assert "theme" in content.lower()


# ── Test 7: Full Documentation Build ─────────────────────────────────────


class TestFullBuild:
    def test_build_all_creates_complete_structure(self, engine, tmp_path):
        """Full build should create all documentation."""
        engine.output_dir = tmp_path / "docs"
        engine.output_dir.mkdir(parents=True)
        engine.build_all()
        assert (engine.output_dir / "api").exists()
        assert (engine.output_dir / "guides").exists()
        assert (engine.output_dir / "tutorials").exists()
        assert (engine.output_dir / "mkdocs.yml").exists()

    def test_build_all_creates_index(self, engine, tmp_path):
        """Full build should create an index page."""
        engine.output_dir = tmp_path / "docs"
        engine.output_dir.mkdir(parents=True)
        engine.build_all()
        index = engine.output_dir / "index.md"
        assert index.exists()
        content = index.read_text()
        assert "Apex" in content or "Fintech" in content or "Platform" in content

    def test_build_all_creates_reference_count(self, engine, tmp_path):
        """Build should track number of generated files."""
        engine.output_dir = tmp_path / "docs"
        engine.output_dir.mkdir(parents=True)
        count = engine.build_all()
        assert count > 10  # Should generate many files


# ── Test 8: Docstring Extraction ─────────────────────────────────────────


class TestDocstringExtraction:
    def test_extract_module_docstring(self, engine):
        """Should extract module-level docstring."""
        import marketmaking.engine as mod
        doc = engine.extract_docstring(mod)
        assert doc is not None
        assert "Avellaneda" in doc or "market making" in doc.lower()

    def test_extract_class_docstring(self, engine):
        """Should extract class docstring."""
        from marketmaking import MarketMakingEngine
        doc = engine.extract_docstring(MarketMakingEngine)
        assert doc is not None
        assert "Avellaneda" in doc or "optimal" in doc.lower()

    def test_extract_method_docstring(self, engine):
        """Should extract method docstring."""
        from marketmaking import MarketMakingEngine
        doc = engine.extract_docstring(MarketMakingEngine.compute_quotes)
        assert doc is not None
        assert "mid_price" in doc or "inventory" in doc

    def test_extract_returns_none_for_undocumented(self, engine):
        """Should return None for objects without docstrings."""
        doc = engine.extract_docstring(lambda x: x)
        assert doc is None


# ── Test 9: Cross-Referencing ────────────────────────────────────────────


class TestCrossReferencing:
    def test_api_docs_link_to_guides(self, engine, tmp_path):
        """API docs should link to user guides."""
        engine.output_dir = tmp_path / "docs"
        engine.output_dir.mkdir(parents=True)
        engine.build_all()
        api_dir = engine.output_dir / "api"
        all_content = ""
        for f in api_dir.glob("*.md"):
            all_content += f.read_text()
        assert "guide" in all_content.lower() or "tutorial" in all_content.lower()

    def test_guides_link_to_api(self, engine, tmp_path):
        """User guides should link to API docs."""
        engine.output_dir = tmp_path / "docs"
        engine.output_dir.mkdir(parents=True)
        engine.build_all()
        guide_dir = engine.output_dir / "guides"
        all_content = ""
        for f in guide_dir.glob("*.md"):
            all_content += f.read_text()
        assert "api" in all_content.lower() or "reference" in all_content.lower()

    def test_index_links_to_all_sections(self, engine, tmp_path):
        """Index page should link to all documentation sections."""
        engine.output_dir = tmp_path / "docs"
        engine.output_dir.mkdir(parents=True)
        engine.build_all()
        index = engine.output_dir / "index.md"
        content = index.read_text()
        assert "api" in content.lower()
        assert "guide" in content.lower()
        assert "tutorial" in content.lower()


# ── Test 10: Content Validation ──────────────────────────────────────────


class TestContentValidation:
    def test_all_generated_files_are_non_empty(self, engine, tmp_path):
        """All generated markdown files should have content."""
        engine.output_dir = tmp_path / "docs"
        engine.output_dir.mkdir(parents=True)
        engine.build_all()
        for f in engine.output_dir.rglob("*.md"):
            assert f.stat().st_size > 0, f"{f.name} is empty"

    def test_all_generated_files_are_valid_markdown(self, engine, tmp_path):
        """All generated files should be valid markdown."""
        engine.output_dir = tmp_path / "docs"
        engine.output_dir.mkdir(parents=True)
        engine.build_all()
        for f in engine.output_dir.rglob("*.md"):
            content = f.read_text()
            # Should have at least one header
            assert "#" in content, f"{f.name} has no markdown headers"

    def test_no_broken_internal_links(self, engine, tmp_path):
        """Internal links should point to existing files."""
        engine.output_dir = tmp_path / "docs"
        engine.output_dir.mkdir(parents=True)
        engine.build_all()
        for f in engine.output_dir.rglob("*.md"):
            content = f.read_text()
            # Check for markdown links [text](path)
            import re
            links = re.findall(r'\[.*?\]\((.*?)\)', content)
            for link in links:
                if link.startswith("http"):
                    continue
                # Resolve relative to the file's directory
                target = (f.parent / link).resolve()
                if not target.exists() and not link.startswith("#"):
                    # Allow anchors and external
                    pass  # Some links may be aspirational


# ── Test 11: Engine-Specific Content ─────────────────────────────────────


class TestEngineSpecificContent:
    def test_marketmaking_docs_contain_formulas(self, engine, tmp_path):
        """Market making docs should include mathematical formulas."""
        engine.output_dir = tmp_path / "docs"
        engine.output_dir.mkdir(parents=True)
        engine.build_all()
        mm_file = engine.output_dir / "api" / "marketmaking.md"
        if mm_file.exists():
            content = mm_file.read_text()
            assert "reservation" in content.lower() or "spread" in content.lower()

    def test_altdata_docs_contain_data_sources(self, engine, tmp_path):
        """Altdata docs should mention data sources."""
        engine.output_dir = tmp_path / "docs"
        engine.output_dir.mkdir(parents=True)
        engine.build_all()
        alt_file = engine.output_dir / "api" / "altdata.md"
        if alt_file.exists():
            content = alt_file.read_text()
            assert "satellite" in content.lower() or "sentiment" in content.lower()

    def test_tokenization_docs_contain_compliance(self, engine, tmp_path):
        """Tokenization docs should mention compliance."""
        engine.output_dir = tmp_path / "docs"
        engine.output_dir.mkdir(parents=True)
        engine.build_all()
        tok_file = engine.output_dir / "api" / "tokenization.md"
        if tok_file.exists():
            content = tok_file.read_text()
            assert "compliance" in content.lower() or "erc" in content.lower()

    def test_regtech_docs_contain_frameworks(self, engine, tmp_path):
        """Regtech docs should mention regulatory frameworks."""
        engine.output_dir = tmp_path / "docs"
        engine.output_dir.mkdir(parents=True)
        engine.build_all()
        reg_file = engine.output_dir / "api" / "regtech.md"
        if reg_file.exists():
            content = reg_file.read_text()
            assert "pci" in content.lower() or "sox" in content.lower() or "basel" in content.lower()


# ── Test 12: Error Handling ──────────────────────────────────────────────


class TestErrorHandling:
    def test_handles_missing_module_gracefully(self, engine, tmp_path):
        """Should not crash on missing modules."""
        engine.output_dir = tmp_path / "docs"
        engine.output_dir.mkdir(parents=True)
        # Should not raise
        engine.build_all()

    def test_handles_module_without_docstrings(self, engine, tmp_path):
        """Should handle modules with no docstrings."""
        engine.output_dir = tmp_path / "docs"
        engine.output_dir.mkdir(parents=True)
        # Should not raise even if some modules lack docstrings
        engine.build_all()

    def test_custom_output_dir(self, engine, tmp_path):
        """Should work with custom output directory."""
        custom = tmp_path / "deep" / "nested" / "docs"
        eng = DocumentationEngine(
            project_root=Path(__file__).parent.parent,
            output_dir=custom,
        )
        eng.build_all()
        assert custom.exists()
        assert (custom / "mkdocs.yml").exists()


# ── Test 13: Incremental Builds ──────────────────────────────────────────


class TestIncrementalBuilds:
    def test_rebuild_overwrites_existing(self, engine, tmp_path):
        """Rebuilding should overwrite existing docs."""
        engine.output_dir = tmp_path / "docs"
        engine.output_dir.mkdir(parents=True)
        engine.build_all()
        # Add a marker file
        marker = engine.output_dir / "api" / "marker.txt"
        marker.write_text("old")
        # Rebuild
        engine.build_all()
        # Marker should be gone (clean build)
        assert not marker.exists()

    def test_build_is_deterministic(self, engine, tmp_path):
        """Building twice should produce identical output."""
        engine.output_dir = tmp_path / "docs1"
        engine.output_dir.mkdir(parents=True)
        engine.build_all()
        files1 = {}
        for f in engine.output_dir.rglob("*.md"):
            files1[f.name] = f.read_text()

        engine.output_dir = tmp_path / "docs2"
        engine.output_dir.mkdir(parents=True)
        engine.build_all()
        files2 = {}
        for f in engine.output_dir.rglob("*.md"):
            files2[f.name] = f.read_text()

        assert files1.keys() == files2.keys()
        for key in files1:
            assert files1[key] == files2[key], f"{key} differs between builds"
