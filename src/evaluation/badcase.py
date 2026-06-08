"""Bad case analysis and tracking.

Logs conversation turns where the answer was unsatisfactory and
provides analysis tools to identify patterns for improvement.
"""

import json
import logging
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class BadCaseRecord:
    """A single bad case entry."""

    id: str
    question: str
    actual_answer: str
    expected_answer: str = ""
    retrieved_sources: List[Dict[str, Any]] = field(default_factory=list)
    category: str = ""  # e.g., "off_topic", "inaccurate", "incomplete", "no_context"
    notes: str = ""
    timestamp: str = field(default_factory=lambda: time.strftime("%Y-%m-%d %H:%M:%S"))
    metadata: Dict[str, Any] = field(default_factory=dict)


class BadCaseAnalyzer:
    """Track, store, and analyze bad cases to guide system improvements.

    Usage:
        analyzer = BadCaseAnalyzer("./data/badcases")
        analyzer.log_bad_case(question, answer, sources, category="inaccurate")
        report = analyzer.analyze()
    """

    CATEGORIES = [
        "off_topic",       # Answer is unrelated to the question
        "inaccurate",      # Answer contains factual errors
        "incomplete",      # Answer misses important information
        "no_context",      # Relevant documents exist but weren't retrieved
        "hallucination",   # Answer includes made-up information
        "format_issue",    # Answer format is problematic
        "other",
    ]

    def __init__(self, data_dir: str = "./data/badcases"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self._records: List[BadCaseRecord] = []
        self._load()

    def log_bad_case(
        self,
        question: str,
        actual_answer: str,
        retrieved_sources: List[Dict[str, Any]],
        expected_answer: str = "",
        category: str = "other",
        notes: str = "",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> BadCaseRecord:
        """Log a new bad case for analysis."""
        import uuid

        record = BadCaseRecord(
            id=str(uuid.uuid4())[:8],
            question=question,
            actual_answer=actual_answer,
            expected_answer=expected_answer,
            retrieved_sources=retrieved_sources,
            category=category if category in self.CATEGORIES else "other",
            notes=notes,
            metadata=metadata or {},
        )

        self._records.append(record)
        self._save()
        logger.info("Bad case logged: id=%s, category=%s", record.id, record.category)
        return record

    def analyze(self) -> Dict[str, Any]:
        """Analyze all bad cases and generate a report.

        Returns:
            Dict with category distribution, common patterns, and recommendations.
        """
        if not self._records:
            return {"message": "No bad cases recorded yet.", "total": 0}

        # Category distribution
        categories: Dict[str, int] = {}
        for r in self._records:
            categories[r.category] = categories.get(r.category, 0) + 1

        # Average source scores (check if retrieval is the issue)
        avg_scores: List[float] = []
        low_retrieval_count = 0
        for r in self._records:
            scores = [s.get("score", 0) for s in r.retrieved_sources]
            if scores:
                avg = sum(scores) / len(scores)
                avg_scores.append(avg)
                if avg < 0.3:
                    low_retrieval_count += 1

        overall_avg_score = sum(avg_scores) / len(avg_scores) if avg_scores else 0.0

        # Generate recommendations
        recommendations: List[str] = []
        if categories.get("no_context", 0) > 0 or low_retrieval_count > len(self._records) * 0.3:
            recommendations.append(
                "检索召回率不足：建议降低 chunk_size 或调高 top_k_retrieval，"
                "或增加 BM25 权重以改善关键词匹配。"
            )
        if categories.get("inaccurate", 0) > len(self._records) * 0.3:
            recommendations.append(
                "答案准确度不足：建议调整 Prompt 模板，强化 '严格基于文档' 的指令；"
                "或考虑更换更强的 LLM 模型。"
            )
        if categories.get("hallucination", 0) > 0:
            recommendations.append(
                "存在模型幻觉：建议提高 rerank 阈值过滤低相关文档，"
                "并在 Prompt 中明确要求 '不确定时请说明'。"
            )
        if categories.get("off_topic", 0) > 0:
            recommendations.append(
                "答非所问：检查文档覆盖范围，考虑添加更多相关文档；"
                "或调整检索权重 (α) 以改善相关性排序。"
            )
        if not recommendations:
            recommendations.append("样本较少，建议继续收集 Bad Case 以发现模式。")

        return {
            "total": len(self._records),
            "category_distribution": categories,
            "average_retrieval_score": round(overall_avg_score, 4),
            "low_retrieval_ratio": round(low_retrieval_count / len(self._records), 2),
            "recommendations": recommendations,
        }

    def export_report(self, output_path: Optional[str] = None) -> str:
        """Export a Markdown analysis report."""
        analysis = self.analyze()
        output_path = output_path or str(self.data_dir / "badcase_report.md")

        lines = [
            "# Bad Case 分析报告",
            f"\n生成时间：{time.strftime('%Y-%m-%d %H:%M:%S')}",
            f"\n总计 Bad Case：{analysis.get('total', 0)} 条",
            "",
            "## 分类分布",
            "",
        ]

        for cat, count in analysis.get("category_distribution", {}).items():
            pct = round(count / analysis["total"] * 100, 1) if analysis["total"] else 0
            lines.append(f"- **{cat}**: {count} ({pct}%)")

        lines.extend([
            "",
            f"## 检索质量",
            f"- 平均检索分数: {analysis.get('average_retrieval_score', 'N/A')}",
            f"- 低召回比例: {analysis.get('low_retrieval_ratio', 'N/A')}",
            "",
            "## 改进建议",
            "",
        ])

        for i, rec in enumerate(analysis.get("recommendations", []), 1):
            lines.append(f"{i}. {rec}")

        lines.extend([
            "",
            "## 详细记录",
            "",
        ])

        for r in self._records:
            lines.append(f"### {r.id} — {r.category}")
            lines.append(f"- 时间: {r.timestamp}")
            lines.append(f"- 问题: {r.question[:200]}")
            lines.append(f"- 实际答案: {r.actual_answer[:300]}")
            if r.expected_answer:
                lines.append(f"- 期望答案: {r.expected_answer[:300]}")
            lines.append(f"- 检索源数量: {len(r.retrieved_sources)}")
            if r.notes:
                lines.append(f"- 备注: {r.notes}")
            lines.append("")

        report = "\n".join(lines)
        Path(output_path).write_text(report, encoding="utf-8")
        logger.info("Report exported to %s", output_path)
        return report

    # ── Persistence ──────────────────────────────────────────────────

    @property
    def _storage_path(self) -> Path:
        return self.data_dir / "badcases.json"

    def _save(self) -> None:
        """Persist records to JSON."""
        data = [asdict(r) for r in self._records]
        self._storage_path.write_text(
            json.dumps(data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def _load(self) -> None:
        """Load records from JSON."""
        if self._storage_path.exists():
            try:
                data = json.loads(self._storage_path.read_text(encoding="utf-8"))
                self._records = [BadCaseRecord(**d) for d in data]
                logger.info("Loaded %d bad case records.", len(self._records))
            except Exception as e:
                logger.warning("Failed to load bad cases: %s", e)
                self._records = []

    def clear(self) -> None:
        """Clear all records."""
        self._records = []
        self._save()
