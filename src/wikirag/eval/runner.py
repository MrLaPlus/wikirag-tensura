import json
from pathlib import Path
from typing import Any, Dict, List, Optional
from rich.console import Console
from rich.table import Table
from wikirag.eval.metrics import compute_hit_rate_at_k, compute_keyword_recall, compute_mrr
from wikirag.retrieval.pipeline import RetrievalPipeline
from wikirag.utils.logging import get_logger

logger = get_logger(__name__)
console = Console()


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

    def run_eval(self, pipeline: RetrievalPipeline, top_k: int = 5) -> Dict[str, float]:
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

        mrr_scores = []
        hit_1_scores = []
        hit_k_scores = []

        for item in dataset:
            q_id = item.get("id", "")
            q_text = item.get("question", "")
            lang = item.get("language", "en")
            exp_entities = item.get("expected_entities", [])

            # Run retrieval
            chunks = pipeline.retrieve(q_text, top_k=top_k)
            retrieved_entities = [c.get("entity", "") for c in chunks]

            h1 = compute_hit_rate_at_k(retrieved_entities, exp_entities, k=1)
            hk = compute_hit_rate_at_k(retrieved_entities, exp_entities, k=top_k)
            mrr = compute_mrr(retrieved_entities, exp_entities)

            hit_1_scores.append(h1)
            hit_k_scores.append(hk)
            mrr_scores.append(mrr)

            table.add_row(
                q_id,
                q_text[:34] + "..." if len(q_text) > 34 else q_text,
                lang,
                f"[green]Yes[/green]" if h1 > 0 else "[red]No[/red]",
                f"[green]Yes[/green]" if hk > 0 else "[red]No[/red]",
                f"{mrr:.2f}",
            )

        console.print(table)

        avg_hit_1 = sum(hit_1_scores) / len(hit_1_scores) if hit_1_scores else 0.0
        avg_hit_k = sum(hit_k_scores) / len(hit_k_scores) if hit_k_scores else 0.0
        avg_mrr = sum(mrr_scores) / len(mrr_scores) if mrr_scores else 0.0

        summary_table = Table(title="Summary Metrics", show_header=True, header_style="bold cyan")
        summary_table.add_column("Metric", style="bold")
        summary_table.add_column("Score")
        summary_table.add_row("Total Questions Evaluated", str(len(dataset)))
        summary_table.add_row("Hit Rate @ 1", f"{avg_hit_1 * 100:.1f}%")
        summary_table.add_row(f"Hit Rate @ {top_k}", f"{avg_hit_k * 100:.1f}%")
        summary_table.add_row("Mean Reciprocal Rank (MRR)", f"{avg_mrr:.3f}")
        console.print(summary_table)

        return {
            "hit_rate_1": avg_hit_1,
            f"hit_rate_{top_k}": avg_hit_k,
            "mrr": avg_mrr,
        }
