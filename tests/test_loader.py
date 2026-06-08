"""Tests for the document loader module."""

import pytest
from pathlib import Path
from src.document.loader import DocumentLoader


class TestDocumentLoader:
    """Test the multi-format document loader."""

    @pytest.fixture
    def loader(self):
        return DocumentLoader()

    @pytest.fixture
    def sample_dir(self, tmp_path: Path):
        """Create temporary test documents."""
        # Markdown file
        md_file = tmp_path / "test.md"
        md_file.write_text("# 测试文档\n\n这是测试内容。\n\n## 第二章\n\n第二段内容。", encoding="utf-8")

        # Text file
        txt_file = tmp_path / "notes.txt"
        txt_file.write_text("这是一段文本笔记。", encoding="utf-8")

        return tmp_path

    def test_load_directory(self, loader, sample_dir):
        docs = loader.load_directory(sample_dir)
        assert len(docs) >= 2

        sources = [d.metadata["file_name"] for d in docs]
        assert "test.md" in sources
        assert "notes.txt" in sources

    def test_load_markdown(self, loader, tmp_path: Path):
        md_file = tmp_path / "sample.md"
        md_file.write_text("# 标题\n\n内容段落。", encoding="utf-8")

        docs = loader.load_markdown(md_file)
        assert len(docs) == 1
        assert "标题" in docs[0].page_content
        assert "内容段落" in docs[0].page_content
        assert docs[0].metadata["format"] == "markdown"
        assert docs[0].metadata["file_name"] == "sample.md"

    def test_load_text(self, loader, tmp_path: Path):
        txt_file = tmp_path / "readme.txt"
        content = "安装说明\n\n1. 安装Python\n2. 安装依赖"
        txt_file.write_text(content, encoding="utf-8")

        docs = loader.load_text(txt_file)
        assert len(docs) == 1
        assert docs[0].page_content == content
        assert docs[0].metadata["format"] == "text"

    def test_unsupported_format(self, loader, tmp_path: Path):
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("a,b,c\n", encoding="utf-8")

        with pytest.raises(ValueError):
            loader.load_file(csv_file)

    def test_file_not_found(self, loader):
        with pytest.raises(FileNotFoundError):
            loader.load_file("/nonexistent/path/file.pdf")

    def test_empty_directory(self, loader, tmp_path: Path):
        docs = loader.load_directory(tmp_path)
        assert docs == []

    def test_metadata_propagation(self, loader, tmp_path: Path):
        md_file = tmp_path / "doc.md"
        md_file.write_text("测试内容。", encoding="utf-8")

        docs = loader.load_markdown(md_file)
        assert "source" in docs[0].metadata
        assert "file_name" in docs[0].metadata
        assert "format" in docs[0].metadata
