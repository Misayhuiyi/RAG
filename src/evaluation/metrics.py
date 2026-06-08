"""Evaluation metrics for the RAG Q&A system.

Computes accuracy, recall, precision, and latency benchmarks
for both retrieval and end-to-end answer quality.
"""

import json
import logging
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class EvalResult:
    """Result of a single evaluation."""

    question: str
    ground_truth: str
    predicted_answer: str
    retrieved_chunks: List[str]
    is_correct: bool  # Human or LLM-judged correctness
    retrieval_recall: float  # fraction of relevant docs retrieved
    latency_ms: float


@dataclass
class MetricsReport:
    """Aggregated evaluation metrics."""

    total_questions: int
    accuracy: float  # correct / total
    avg_recall: float
    avg_precision: float
    avg_latency_ms: float
    p50_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float
    details: List[EvalResult]


class MetricsCalculator:
    """Calculate evaluation metrics for the RAG system.

    Usage:
        calc = MetricsCalculator()
        # Load ground truth from test set
        calc.load_test_set("./data/test_questions.json")
        # After running queries, compute metrics
        report = calc.compute_metrics(results)
    """

    def __init__(self):
        self._test_set: List[Dict[str, Any]] = []
        self._results: List[EvalResult] = []

    def load_test_set(self, path: str | Path) -> List[Dict[str, Any]]:
        """Load a test question set from JSON.

        Expected format:
        [
            {
                "question": "...",
                "answer": "...",           # expected answer
                "relevant_docs": ["doc1.md", "doc2.md"],  # optional
                "keywords": ["关键词1", "关键词2"]         # optional
            },
            ...
        ]
        """
        path = Path(path)
        if not path.exists():
            logger.warning("Test set not found: %s", path)
            return []

        with open(path, "r", encoding="utf-8") as f:
            self._test_set = json.load(f)

        logger.info("Loaded %d test questions from %s", len(self._test_set), path)
        return self._test_set

    def add_result(self, result: EvalResult) -> None:
        """Add a single evaluation result."""
        self._results.append(result)

    def compute_metrics(self) -> MetricsReport:
        """Compute aggregate metrics from all results."""
        if not self._results:
            return MetricsReport(
                total_questions=0, accuracy=0.0, avg_recall=0.0,
                avg_precision=0.0, avg_latency_ms=0.0,
                p50_latency_ms=0.0, p95_latency_ms=0.0, p99_latency_ms=0.0,
                details=[],
            )

        total = len(self._results)
        correct = sum(1 for r in self._results if r.is_correct)
        accuracy = correct / total if total > 0 else 0.0
        avg_recall = sum(r.retrieval_recall for r in self._results) / total
        latencies = sorted([r.latency_ms for r in self._results])

        return MetricsReport(
            total_questions=total,
            accuracy=round(accuracy, 4),
            avg_recall=round(avg_recall, 4),
            avg_precision=0.0,  # Requires annotation of relevant chunks
            avg_latency_ms=round(statistics.mean(latencies), 2),
            p50_latency_ms=round(self._percentile(latencies, 50), 2),
            p95_latency_ms=round(self._percentile(latencies, 95), 2),
            p99_latency_ms=round(self._percentile(latencies, 99), 2),
            details=self._results,
        )

    async def evaluate_pipeline(
        self,
        query_pipeline,  # QueryPipeline
        llm_judge: bool = True,
    ) -> MetricsReport:
        """Run the full test set through the query pipeline and evaluate.

        Args:
            query_pipeline: An initialized QueryPipeline instance.
            llm_judge: If True, use an LLM to judge answer correctness.
                       Otherwise, use keyword matching.
        """
        self._results = []

        for item in self._test_set:
            question = item["question"]
            expected = item.get("answer", "")
            relevant_docs = set(item.get("relevant_docs", []))

            # Time the query
            start = time.time()
            try:
                response = await query_pipeline.answer(
                    question=question,
                    session_id="eval",
                )
                latency_ms = (time.time() - start) * 1000

                predicted = response["answer"]
                sources = response["sources"]

                # Calculate retrieval recall
                retrieved_names = set(s.get("document_name", "") for s in sources)
                recall = (
                    len(retrieved_names & relevant_docs) / len(relevant_docs)
                    if relevant_docs else 1.0
                )

                # Judge correctness
                if llm_judge:
                    is_correct = await self._llm_judge(question, predicted, expected, query_pipeline)
                else:
                    is_correct = self._keyword_judge(predicted, expected, item.get("keywords", []))

                self._results.append(EvalResult(
                    question=question,
                    ground_truth=expected,
                    predicted_answer=predicted,
                    retrieved_chunks=[s.get("content", "") for s in sources],
                    is_correct=is_correct,
                    retrieval_recall=recall,
                    latency_ms=latency_ms,
                ))

            except Exception as e:
                logger.error("Evaluation error for question '%s': %s", question[:50], e)
                self._results.append(EvalResult(
                    question=question,
                    ground_truth=expected,
                    predicted_answer=f"ERROR: {e}",
                    retrieved_chunks=[],
                    is_correct=False,
                    retrieval_recall=0.0,
                    latency_ms=(time.time() - start) * 1000,
                ))

        return self.compute_metrics()

    # ── Judging ──────────────────────────────────────────────────────

    @staticmethod
    def _keyword_judge(
        predicted: str,
        expected: str,
        keywords: List[str],
    ) -> bool:
        """Simple keyword-match judge.

        Returns True if all keywords are found in the predicted answer.
        """
        if not keywords:
            # Without keywords, check if predicted contains expected text
            return expected.lower() in predicted.lower() if expected else True

        return all(kw.lower() in predicted.lower() for kw in keywords)

    async def _llm_judge(
        self,
        question: str,
        predicted: str,
        expected: str,
        query_pipeline,
    ) -> bool:
        """Use the pipeline's LLM to judge if the predicted answer satisfies the question."""
        from ..llm.base import Message

        judge_prompt = (
            f"你是一个评估助手。判断以下预测答案是否正确回答了用户问题。\n\n"
            f"## 用户问题\n{question}\n\n"
            f"## 期望答案（参考标准）\n{expected}\n\n"
            f"## 预测答案\n{predicted}\n\n"
            f"请判断预测答案是否准确、完整地回答了用户问题，且不包含事实错误。\n"
            f"如果预测答案与期望答案内容一致（包括逻辑等价的不同表述），则回答「正确」。\n"
            f"如果预测答案遗漏关键信息、包含事实错误或答非所问，则回答「错误」。\n\n"
            f"只需回复「正确」或「错误」，不要额外解释。"
        )

        try:
            response = await query_pipeline.llm.generate(
                prompt=judge_prompt,
                context=[],
                history=[],
                system_prompt="你是一个严格的答案评估助手。只输出「正确」或「错误」。",
            )
            answer = response.answer.strip()
            return "正确" in answer and "错误" not in answer
        except Exception as e:
            logger.warning("LLM judge failed, falling back to keyword judge: %s", e)
            # Fallback to keyword matching
            if not expected:
                return True
            return expected.lower() in predicted.lower()

    # ── Benchmarking ─────────────────────────────────────────────────

    @staticmethod
    async def benchmark_latency(
        query_pipeline,
        queries: List[str],
        n_runs: int = 3,
    ) -> Dict[str, float]:
        """Benchmark query latency across multiple runs.

        Returns:
            Dict with avg, min, max, p50, p95, p99 latency in milliseconds.
        """
        all_latencies: List[float] = []

        for query in queries:
            for _ in range(n_runs):
                start = time.time()
                await query_pipeline.answer(question=query, session_id="bench")
                elapsed = (time.time() - start) * 1000
                all_latencies.append(elapsed)

        sorted_lats = sorted(all_latencies)
        return {
            "avg_ms": round(statistics.mean(all_latencies), 2),
            "min_ms": round(min(all_latencies), 2),
            "max_ms": round(max(all_latencies), 2),
            "p50_ms": round(MetricsCalculator._percentile(sorted_lats, 50), 2),
            "p95_ms": round(MetricsCalculator._percentile(sorted_lats, 95), 2),
            "p99_ms": round(MetricsCalculator._percentile(sorted_lats, 99), 2),
            "total_measurements": len(all_latencies),
        }

    @staticmethod
    def _percentile(sorted_data: List[float], percentile: float) -> float:
        """Compute the nth percentile of sorted data."""
        if not sorted_data:
            return 0.0
        k = (len(sorted_data) - 1) * percentile / 100.0
        f = int(k)
        c = f + 1 if f + 1 < len(sorted_data) else f
        return sorted_data[f] + (sorted_data[c] - sorted_data[f]) * (k - f)
