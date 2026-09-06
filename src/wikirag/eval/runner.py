import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional
from rich.console import Console
from rich.table import Table
from wikirag.eval.metrics import compute_hit_rate_at_k, compute_keyword_recall, compute_mrr
from wikirag.retrieval.pipeline import RetrievalPipeline
from wikirag.utils.logging import get_logger

logger = get_logger(__name__)
console = Console()


def _display_text(value: str) -> str:
    """Keep CLI evaluation usable on legacy Windows code pages."""
    encoding = getattr(sys.stdout, "encoding", None) or "utf-8"
    try:
        value.encode(encoding)
        return value
    except UnicodeEncodeError:
        return value.encode(encoding, errors="replace").decode(encoding, errors="replace")


class EvalRunner:
    """Automated benchmark harness evaluating retrieval quality against Golden Q&A pairs."""

    def __init__(self, golden_path: str = "eval/golden_qa.json"):
        self.golden_path = Path(golden_path)

    def load_golden_set(self) -> List[Dict[str, Any]]:
        if not self.golden_path.exists():
            logger.warning(f"Golden dataset not found at {self.golden_path}")
            return []
        with open(self.golden_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def run_eval(self, pipeline: RetrievalPipeline, top_k: int = 5) -> Dict[str, Any]:
        dataset = self.load_golden_set()
        if not dataset:
            return {"hit_rate_1": 0.0, "hit_rate_5": 0.0, "mrr": 0.0}

        table = Table(title="WikiRAG Evaluation Results", show_header=True, header_style="bold green")
        table.add_column("ID", style="dim", width=6)
        table.add_column("Question", width=36)
        table.add_column("Lang", width=6)
        table.add_column("Hit@1", width=8)
        table.add_column(f"Hit@{top_k}", width=8)
        table.add_column("MRR", width=8)

        modes = {"dense": {"hit_1": [], "hit_k": [], "mrr": [], "keyword_recall": []},
                 "hybrid": {"hit_1": [], "hit_k": [], "mrr": [], "keyword_recall": []}}

        for item in dataset:
            q_id = item.get("id", "")
            q_text = item.get("question", "")
            lang = item.get("language", "en")
            exp_entities = item.get("expected_entities", [])

            # Compare the baseline leg with the BM25-enabled hybrid leg.
            # Both retain the existing graph/structured expansion used by the app.
            for mode, use_bm25 in (("dense", False), ("hybrid", True)):
                chunks = pipeline.retrieve(q_text, top_k=top_k, enable_bm25=use_bm25)
                retrieved_entities = [c.get("entity", "") for c in chunks]
                h1 = compute_hit_rate_at_k(retrieved_entities, exp_entities, k=1)
                hk = compute_hit_rate_at_k(retrieved_entities, exp_entities, k=top_k)
                mrr = compute_mrr(retrieved_entities, exp_entities)
                context = " ".join(str(c.get("chunk_text", "")) for c in chunks)
                keyword_recall = compute_keyword_recall(context, item.get("expected_answer_keywords", []))
                modes[mode]["hit_1"].append(h1)
                modes[mode]["hit_k"].append(hk)
                modes[mode]["mrr"].append(mrr)
                modes[mode]["keyword_recall"].append(keyword_recall)

            h1 = modes["hybrid"]["hit_1"][-1]
            hk = modes["hybrid"]["hit_k"][-1]
            mrr = modes["hybrid"]["mrr"][-1]

            table.add_row(
                q_id,
                _display_text(q_text[:34] + "..." if len(q_text) > 34 else q_text),
                lang,
                f"[green]Yes[/green]" if h1 > 0 else "[red]No[/red]",
                f"[green]Yes[/green]" if hk > 0 else "[red]No[/red]",
                f"{mrr:.2f}",
            )

        console.print(table)

        def averages(values: Dict[str, List[float]]) -> Dict[str, float]:
            return {key: (sum(items) / len(items) if items else 0.0) for key, items in values.items()}

        dense = averages(modes["dense"])
        hybrid = averages(modes["hybrid"])

        summary_table = Table(title="Summary Metrics", show_header=True, header_style="bold cyan")
        summary_table.add_column("Metric", style="bold")
        summary_table.add_column("Score")
        summary_table.add_row("Total Questions Evaluated", str(len(dataset)))
        summary_table.add_row("Hybrid Hit Rate @ 1", f"{hybrid['hit_1'] * 100:.1f}%")
        summary_table.add_row(f"Hybrid Hit Rate @ {top_k}", f"{hybrid['hit_k'] * 100:.1f}%")
        summary_table.add_row("Hybrid MRR", f"{hybrid['mrr']:.3f}")
        summary_table.add_row("Hybrid Context Keyword Recall", f"{hybrid['keyword_recall'] * 100:.1f}%")
        console.print(summary_table)

        return {
            "hit_rate_1": hybrid["hit_1"],
            f"hit_rate_{top_k}": hybrid["hit_k"],
            "mrr": hybrid["mrr"],
            "context_keyword_recall": hybrid["keyword_recall"],
            "comparison": {
                "dense": dense,
                "hybrid": hybrid,
                "delta": {key: hybrid[key] - dense[key] for key in hybrid},
            },
            "answer_quality": {"available": False, "reason": "ยังไม่ได้สร้างคำตอบ LLM ใน Evaluation รอบนี้"},
            "citation_support": {"available": True, "definition": "คำนวณจากการพบ expected_entities ในผลค้นหา ไม่ใช่การตรวจเนื้อหา Citation แบบ LLM"},
        }
