import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from config import cfg
from consistency_hybrid import ConsistencyBuilderHybrid
from diagnose_validation import signature_from_config, threshold_scan


class ValidationDiagnosticTests(unittest.TestCase):
    def test_signature_matches_real_constructor_without_model_downloads(self):
        with tempfile.TemporaryDirectory() as directory:
            Path(directory, "modules.json").write_text("[]", encoding="utf-8")
            modules = {
                "pycorenlp": SimpleNamespace(StanfordCoreNLP=MagicMock()),
                "sentence_transformers": SimpleNamespace(SentenceTransformer=MagicMock()),
                "transformers": SimpleNamespace(
                    AutoTokenizer=MagicMock(), AutoModelForSequenceClassification=MagicMock()),
            }
            for use_nli in (False, True):
                config = replace(cfg, SBERT_MODEL=directory, NLI_MODEL=directory, USE_NLI=use_nli)
                with patch.dict("sys.modules", modules):
                    builder = ConsistencyBuilderHybrid(
                        sbert_model=directory, nli_model=directory, use_nli=use_nli,
                        nli_contradiction_id=2, device="cpu",
                    )
                    try:
                        self.assertEqual(builder.cache_signature, signature_from_config(config))
                    finally:
                        builder.close()

    def test_threshold_selection_uses_strict_comparison_and_reference_tiebreak(self):
        rows, selected = threshold_scan([0, 0, 1, 1], [0.1, 0.4, 0.45, 0.8], 0.55)
        self.assertEqual(selected["threshold"], 0.44)
        self.assertEqual(selected["macro_f1"], 1.0)
        at_boundary = next(r for r in rows if r["threshold"] == 0.45)
        self.assertEqual(at_boundary["fake_recall"], 0.5)
        _, tied = threshold_scan([0, 1], [0.01, 0.99], 0.55)
        self.assertEqual(tied["threshold"], 0.55)


if __name__ == "__main__":
    unittest.main()
