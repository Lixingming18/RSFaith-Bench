from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from rsfaith_bench.answer import gold_answer  # noqa: E402
from rsfaith_bench.data import item_id, load_items  # noqa: E402
from rsfaith_bench.evaluate import evaluate_item  # noqa: E402
from rsfaith_bench.report import summarize_records, write_summary_csv  # noqa: E402


class SmokePipelineTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.items = load_items(ROOT / "RSFaith-Bench_subset")

    def test_subset_loads(self) -> None:
        self.assertEqual(len(self.items), 600)
        self.assertEqual(len({item["subcategory"] for item in self.items}), 12)

    def test_small_evaluation_pipeline(self) -> None:
        rows = []
        for item in self.items[:6]:
            prediction = {
                "question_id": item_id(item),
                "model_name": "smoke-gold-support",
                "response": f"Evidence: Smoke-test response.\nAnswer: {gold_answer(item)}",
            }
            claim_record = {
                "question_id": item_id(item),
                "model_name": "smoke-gold-support",
                "claims": item.get("support", []),
            }
            row = evaluate_item(item, prediction, claim_record=claim_record)
            rows.append(row)
            self.assertIn("metrics", row)
            self.assertIn("verification", row)
            self.assertIn("unmapped_claims", row)
            self.assertTrue(row["metrics"]["aa"])

        summary = summarize_records(rows, group_fields=("model_name",))
        self.assertEqual(len(summary), 1)
        for key in ("aa", "cp", "fa", "C-CUR", "M-CUR"):
            self.assertIn(key, summary[0])

    def test_summary_csv_writes(self) -> None:
        item = self.items[0]
        row = evaluate_item(
            item,
            {
                "question_id": item_id(item),
                "model_name": "smoke-gold-support",
                "response": f"Evidence: Smoke-test response.\nAnswer: {gold_answer(item)}",
            },
            claim_record={
                "question_id": item_id(item),
                "model_name": "smoke-gold-support",
                "claims": item.get("support", []),
            },
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "summary.csv"
            write_summary_csv([row], path, group_fields=("model_name",))
            text = path.read_text(encoding="utf-8")
        self.assertIn("model_name,n,aa,cp,fa,C-CUR,M-CUR", text)
        self.assertIn("smoke-gold-support", text)


if __name__ == "__main__":
    unittest.main()
