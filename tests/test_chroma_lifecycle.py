"""Regression tests for ChromaDB PersistentClient resource cleanup.

Issue #92: clients were not closed, causing PermissionError on Windows during
TemporaryDirectory cleanup of chroma.sqlite3 / data_level0.bin.
"""

import os
import sys
import tempfile
import unittest
from unittest.mock import MagicMock, patch


class TestChromaLifecycle(unittest.TestCase):
    def test_ingest_closes_client_even_on_success(self) -> None:
        from src.regbot.ingestion import ingest_policy_file

        mock_client = MagicMock()
        mock_collection = MagicMock()
        mock_client.get_or_create_collection.return_value = mock_collection

        mock_chromadb = MagicMock()
        mock_chromadb.PersistentClient.return_value = mock_client

        with tempfile.TemporaryDirectory() as store:
            path = os.path.join(store, "policy.txt")
            with open(path, "w", encoding="utf-8") as f:
                f.write(
                    "This policy requires informed consent for genomic data sharing and subsequent secondary use of the samples in research studies that have been approved by an ethics committee."
                )

            with (
                patch.dict(sys.modules, {"chromadb": mock_chromadb}),
                patch("src.regbot.ingestion.chromadb_settings", return_value=MagicMock()),
                patch("src.regbot.ingestion.load_sentence_transformer") as mock_model,
                patch(
                    "src.regbot.ingestion.load_document_pages",
                    return_value=[
                        (
                            "This policy requires informed consent for genomic data sharing and subsequent secondary use of the samples in research studies that have been approved by an ethics committee.",
                            1,
                        )
                    ],
                ),
            ):
                mock_encoder = MagicMock()
                import numpy as np

                mock_encoder.encode.return_value = np.array([[0.1] * 384])
                mock_model.return_value = mock_encoder

                n = ingest_policy_file(path, store)
                self.assertGreaterEqual(n, 0)
                mock_client.close.assert_called_once()

    def test_ingest_closes_client_on_error_path(self) -> None:
        from src.regbot.ingestion import ingest_policy_file

        mock_client = MagicMock()
        mock_chromadb = MagicMock()
        mock_chromadb.PersistentClient.return_value = mock_client

        with tempfile.TemporaryDirectory() as store:
            path = os.path.join(store, "empty.txt")
            with open(path, "w", encoding="utf-8") as f:
                f.write("")

            with (
                patch.dict(sys.modules, {"chromadb": mock_chromadb}),
                patch("src.regbot.ingestion.chromadb_settings", return_value=MagicMock()),
                patch("src.regbot.ingestion.load_document_pages", return_value=[("", 1)]),
            ):
                with self.assertRaises(ValueError):
                    ingest_policy_file(path, store)
                mock_client.close.assert_called_once()

    def test_retriever_close_releases_client(self) -> None:
        from src.regbot.retrieval import HybridRetriever

        mock_client = MagicMock()
        mock_collection = MagicMock()
        mock_client.get_collection.return_value = mock_collection
        mock_chromadb = MagicMock()
        mock_chromadb.PersistentClient.return_value = mock_client

        with tempfile.TemporaryDirectory() as store:
            chroma = os.path.join(store, "chroma")
            os.makedirs(chroma)
            r = HybridRetriever(store)
            with (
                patch.dict(sys.modules, {"chromadb": mock_chromadb}),
                patch("src.regbot.retrieval.chromadb_settings", return_value=MagicMock()),
            ):
                r._ensure_loaded()
                self.assertIsNotNone(r._client)
                r.close()
                mock_client.close.assert_called_once()
                self.assertIsNone(r._client)
                self.assertIsNone(r._collection)


if __name__ == "__main__":
    unittest.main()
