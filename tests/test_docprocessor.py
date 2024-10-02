# SPDX-License-Identifier: Apache-2.0

# Standard
from typing import Any, Dict, Union
from unittest.mock import MagicMock, patch
import os

# Third Party
import pytest
import yaml
from docling.document_converter import DocumentConverter

# First Party
from instructlab.sdg.utils import docprocessor
from instructlab.sdg.utils import taxonomy

# Local
from .testdata import testdata

TEST_DATA_DIR = os.path.join(os.path.dirname(__file__), "testdata")

def load_test_skills(skills_file_path) -> Union[Dict[str, Any], None]:
    with open(skills_file_path, "r", encoding="utf-8") as skills_file:
        return yaml.safe_load(skills_file)

class TestChunking:
    """Test collection in instructlab.utils.chunking."""

    def test_chunk_docs_wc_exceeds_ctx_window(self):
        with pytest.raises(ValueError) as exc:
            docprocessor.chunk_markdowns(
                documents=testdata.documents,
                chunk_word_count=1000,
                server_ctx_size=1034,
            )
        assert (
            "Given word count (1000) per doc will exceed the server context window size (1034)"
            in str(exc.value)
        )

    def test_chunk_docs_chunk_overlap_error(self):
        with pytest.raises(ValueError) as exc:
            docprocessor.chunk_markdowns(
                documents=testdata.documents,
                chunk_word_count=5,
                server_ctx_size=1034,
            )
        assert (
            "Got a larger chunk overlap (100) than chunk size (24), should be smaller"
            in str(exc.value)
        )

    def test_chunk_docs_long_lines(self):
        chunk_words = 50
        chunks = docprocessor.chunk_markdowns(
            documents=testdata.long_line_documents,
            chunk_word_count=chunk_words,
            server_ctx_size=4096,
        )
        max_tokens = docprocessor._num_tokens_from_words(chunk_words)
        max_chars = docprocessor._num_chars_from_tokens(max_tokens)
        max_chars += docprocessor._DEFAULT_CHUNK_OVERLAP  # add in the chunk overlap
        max_chars += 50  # and a bit extra for some really long words
        for chunk in chunks:
            assert len(chunk) <= max_chars

class TestMarkdownDocuments:
    @pytest.fixture(autouse=True)
    def _init_taxonomy(self, taxonomy_dir):
        self.taxonomy = taxonomy_dir

    def test_chunk_markdown_docs(self, tmp_path):
        test_valid_knowledge_skill_file = os.path.join(
            TEST_DATA_DIR, "test_valid_knowledge_skill.yaml"
        )
        tracked_knowledge_file = os.path.join("knowledge", "tracked", "qna.yaml")
        test_valid_knowledge_skill = load_test_skills(test_valid_knowledge_skill_file)
        self.taxonomy.add_tracked(
            tracked_knowledge_file, test_valid_knowledge_skill
        )
        taxonomy_base = "empty"
        leaf_nodes = taxonomy.read_taxonomy_leaf_nodes(
            self.taxonomy.root, taxonomy_base, None, tmp_path
        )
        assert len(leaf_nodes) == 1
        for leaf_node in leaf_nodes.values():
            chunks = docprocessor.chunk_documents(
                leaf_node,
                server_ctx_size=4096,
                chunk_word_count=50,
                output_dir=tmp_path,
                model_name=None,
            )
            assert len(chunks) > 0
            # TODO: more meaningful assertions here to validate chunks

# TODO: The real convert takes quite a while - too long for unit
# tests. Come up with some mocks to speed things up.
#
# def _fake_convert(self, input: DocumentConversionInput) -> Iterable[ConversionResult]:
#     return None

# @patch.object(DocumentConverter, "convert", return_value=[])
class TestPdfDocuments:
    @pytest.fixture(autouse=True)
    def _init_taxonomy(self, taxonomy_dir):
        self.taxonomy = taxonomy_dir

    def test_chunk_pdf_docs(self, tmp_path):
        test_valid_knowledge_skill_file = os.path.join(
            TEST_DATA_DIR, "test_valid_pdf_knowledge_skill.yaml"
        )
        tracked_knowledge_file = os.path.join("knowledge", "tracked", "qna.yaml")
        test_valid_knowledge_skill = load_test_skills(test_valid_knowledge_skill_file)
        self.taxonomy.add_tracked(
            tracked_knowledge_file, test_valid_knowledge_skill
        )
        taxonomy_base = "empty"
        leaf_nodes = taxonomy.read_taxonomy_leaf_nodes(
            self.taxonomy.root, taxonomy_base, None, tmp_path
        )
        assert len(leaf_nodes) == 1
        for leaf_node in leaf_nodes.values():
            chunks = docprocessor.chunk_documents(
                leaf_node,
                server_ctx_size=4096,
                chunk_word_count=50,
                output_dir=tmp_path,
                model_name=None,
            )
            assert len(chunks) > 0
            # TODO: more meaningful assertions here to validate chunks
