"""
api/services/response_adapter.py

Converts OrchestratorAgent's FinalResponse into the AnalysisResponse shape
that the frontend's useAnalysisStream hook consumes.

Design principles
─────────────────
• This adapter NEVER calls an LLM.  All transformations are deterministic
  (zero latency, no network).
• Sentiment is sourced from the structured AgentSentiment field produced by
  each agent — not from word-count heuristics on the narrative text.
• DataFrames are converted to a frontend-friendly table dict here; they never
  cross the HTTP boundary as raw objects.
• All extraction helpers are private static methods so they are independently
  unit-testable without instantiating the class.
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional

import pandas as pd

from core.agents.models.base_agent_models import AgentSentiment
from core.agents.models.news_agent_models import CitedSource
from core.logger import get_logger

logger = get_logger(__name__)

# Neutral sentinel used when an agent produced no structured sentiment
_NEUTRAL_SENTIMENT = AgentSentiment(score=50, label="NEUTRAL", rationale="")


class ResponseAdapter:
    """
    Stateless adapter — safe to instantiate per-request or as a singleton.
    """

    # ── Public entry points ───────────────────────────────────────────────────

    def to_agent_analysis(
        self,
        *,
        agent_name: str,
        analysis_text: str,
        sources: List[CitedSource],
        financial_data: Optional[pd.DataFrame],
        ticker: Optional[str],
        sentiment: Optional[AgentSentiment],
    ) -> dict:
        """
        Map one agent's output to the AgentAnalysis shape consumed by the frontend.
        """
        if agent_name == "news_agent":
            return self._news_agent(
                text=analysis_text,
                sources=sources,
                sentiment=sentiment or _NEUTRAL_SENTIMENT,
            )
        if agent_name in ("fundamentals_agent", "fundamentals"):
            return self._fundamentals_agent(
                text=analysis_text,
                df=financial_data,
                ticker=ticker,
                sentiment=sentiment or _NEUTRAL_SENTIMENT,
            )
        # Fallback for any future agent
        return {
            "id": agent_name,
            "name": agent_name,
            "icon": "other",
            "category": "Analysis",
            "recentCatalyst": {"title": "", "description": "", "timeAgo": ""},
            "sentiment": self._sentiment_dict(sentiment or _NEUTRAL_SENTIMENT),
            "fullReport": analysis_text,
        }

    def to_summary(
        self,
        raw_summary: str,
        agent_analyses: Dict[str, str],
        agent_sentiments: Dict[str, Optional[AgentSentiment]],
    ) -> dict:
        """
        Produce the SummaryOfFindings shape from the orchestrator's final synthesis.

        The overall verdict label is derived from the average of agent sentiment
        scores rather than heuristic keyword matching.
        """
        verdict_label, verdict_desc = self._derive_verdict(
            agent_sentiments, raw_summary
        )
        consensus = self._build_consensus(agent_analyses)
        return {
            "coreNarrative": self._first_n_words(raw_summary, 60),
            "agentConsensus": consensus,
            "verdict": {"label": verdict_label, "description": verdict_desc},
        }

    # ── News agent builder ────────────────────────────────────────────────────

    def _news_agent(
        self,
        text: str,
        sources: List[CitedSource],
        sentiment: AgentSentiment,
    ) -> dict:
        return {
            "id": "news",
            "name": "News Analysis Agent",
            "icon": "news",
            "category": "Intelligence Unit",
            "recentCatalyst": self._extract_catalyst(text),
            "sentiment": self._sentiment_dict(sentiment),
            "fullReport": text,
            "references": [
                {
                    "id": s.source_id,
                    "title": s.title,
                    "url": s.url,
                    "source": self._extract_domain(s.url),
                }
                for s in sources
            ],
        }

    # ── Fundamentals agent builder ────────────────────────────────────────────

    def _fundamentals_agent(
        self,
        text: str,
        df: Optional[pd.DataFrame],
        ticker: Optional[str],
        sentiment: AgentSentiment,
    ) -> dict:
        return {
            "id": "fundamental",
            "name": "Fundamental Agent",
            "icon": "analytics",
            "category": "Financial Lab",
            "recentCatalyst": {"title": "", "description": "", "timeAgo": ""},
            "sentiment": self._sentiment_dict(sentiment),
            "metrics": self._extract_key_metrics(df),
            "quote": self._extract_first_sentence(text),
            "fullReport": text,
            "tableData": self._df_to_table(df) if df is not None else None,
        }

    # ── Sentiment helpers ─────────────────────────────────────────────────────

    @staticmethod
    def _sentiment_dict(s: AgentSentiment) -> dict:
        return {
            "score": s.score,
            "label": f"{s.label} ({s.score}%)",
            "rationale": s.rationale,
        }

    @staticmethod
    def _derive_verdict(
        agent_sentiments: Dict[str, Optional[AgentSentiment]],
        summary_text: str,
    ) -> tuple[str, str]:
        """
        Derive an overall verdict from the average of agent sentiment scores.

        Falls back to NEUTRAL with the last summary sentence as description
        when no structured sentiment is available.
        """
        scores = [s.score for s in agent_sentiments.values() if s is not None]
        if not scores:
            return "NEUTRAL", summary_text[-200:].strip() if summary_text else ""

        avg = sum(scores) / len(scores)
        if avg >= 75:
            label = "STRONG BUY"
        elif avg >= 60:
            label = "BUY"
        elif avg >= 40:
            label = "NEUTRAL"
        elif avg >= 25:
            label = "SELL"
        else:
            label = "STRONG SELL"

        # Use last meaningful sentence of the summary as the verdict description
        sentences = re.split(r"(?<=[.!?])\s+", summary_text.strip())
        desc = sentences[-1][:200] if sentences else ""
        return label, desc

    # ── Catalyst extraction ───────────────────────────────────────────────────

    @staticmethod
    def _extract_catalyst(text: str) -> dict:
        """
        Pull the most event-rich sentence from the analysis as the 'recent catalyst'.
        """
        sentences = re.split(r"(?<=[.!?])\s+", text)
        keywords = [
            "earnings",
            "revenue",
            "announc",
            "launch",
            "acqui",
            "partner",
            "guidance",
            "appoint",
            "expand",
            "cut",
            "beat",
            "miss",
        ]
        for sentence in sentences[:15]:
            if any(kw in sentence.lower() for kw in keywords):
                title = sentence[:80].strip().rstrip(".")
                return {
                    "title": title,
                    "description": sentence.strip(),
                    "timeAgo": "RECENT",
                }
        return {
            "title": "Market Update",
            "description": sentences[0][:120] if sentences else "",
            "timeAgo": "RECENT",
        }

    # ── Key metrics from DataFrame ────────────────────────────────────────────

    @staticmethod
    def _extract_key_metrics(df: Optional[pd.DataFrame]) -> List[dict]:
        """
        Extract up to 4 display metrics from the most recent column of the DataFrame.
        Prefers computed ratio rows over raw EDGAR rows for readability.
        """
        if df is None or df.empty:
            return []

        priority_rows = [
            ("gross_margin", "GROSS MARGIN", "pct"),
            ("net_margin", "NET MARGIN", "pct"),
            ("operating_margin", "OP MARGIN", "pct"),
            ("debt_to_equity", "D/E RATIO", "ratio"),
            ("current_ratio", "CURRENT RATIO", "ratio"),
            ("Revenues", "REVENUE", "currency"),
            ("NetIncomeLoss", "NET INCOME", "currency"),
            ("stock_price", "STOCK PRICE", "price"),
        ]

        metrics: List[dict] = []
        try:
            last_col = df.columns[-1]
        except IndexError:
            return []

        for row_label, display_label, fmt in priority_rows:
            if row_label not in df.index:
                continue
            val = df.loc[row_label, last_col]
            if not pd.notna(val):
                continue
            val = float(val)
            if fmt == "pct":
                # Values < 1 are already decimals (e.g. 0.43); > 1 are already %
                formatted = f"{val:.1%}" if abs(val) <= 1 else f"{val:.1f}%"
            elif fmt == "ratio":
                formatted = f"{val:.2f}x"
            elif fmt == "price":
                formatted = f"${val:,.2f}"
            else:  # currency
                if abs(val) >= 1e12:
                    formatted = f"${val/1e12:.2f}T"
                elif abs(val) >= 1e9:
                    formatted = f"${val/1e9:.2f}B"
                elif abs(val) >= 1e6:
                    formatted = f"${val/1e6:.0f}M"
                else:
                    formatted = f"${val:,.0f}"
            metrics.append({"label": display_label, "value": formatted})
            if len(metrics) >= 4:
                break

        return metrics

    # ── DataFrame → table dict ────────────────────────────────────────────────

    @staticmethod
    def _df_to_table(df: pd.DataFrame) -> Optional[dict]:
        """
        Convert the financial DataFrame into a {title, headers, rows} table dict.

        Takes the 5 most recent period columns and at most 8 rows to keep
        the frontend table manageable.
        """
        if df is None or df.empty:
            return None

        display_df = df.iloc[:8, -5:]
        headers = ["Metric"] + [str(c)[:10] for c in display_df.columns]
        rows: List[List[str]] = []

        for label, row in display_df.iterrows():
            formatted_row = [str(label)]
            for val in row:
                if pd.isna(val):
                    formatted_row.append("—")
                elif abs(val) >= 1e9:
                    formatted_row.append(f"{val / 1e9:.2f}B")
                elif abs(val) >= 1e6:
                    formatted_row.append(f"{val / 1e6:.1f}M")
                elif abs(val) < 10:
                    # Likely a ratio or margin
                    formatted_row.append(f"{val:.4f}")
                else:
                    formatted_row.append(f"{val:.2f}")
            rows.append(formatted_row)

        return {"title": "Financial Data", "headers": headers, "rows": rows}

    # ── Generic text helpers ──────────────────────────────────────────────────

    @staticmethod
    def _first_n_words(text: str, n: int) -> str:
        words = text.split()
        return " ".join(words[:n]) + ("..." if len(words) > n else "")

    @staticmethod
    def _extract_first_sentence(text: str) -> str:
        sentences = re.split(r"(?<=[.!?])\s+", text.strip())
        return sentences[0][:200] if sentences else ""

    @staticmethod
    def _extract_domain(url: str) -> str:
        match = re.search(r"https?://(?:www\.)?([^/]+)", url or "")
        return match.group(1) if match else "Source"

    @staticmethod
    def _build_consensus(agent_analyses: Dict[str, str]) -> List[dict]:
        consensus: List[dict] = []
        if "news_agent" in agent_analyses:
            consensus.append(
                {
                    "title": "News Sentiment",
                    "description": "Recent media coverage and event catalysts assessed.",
                    "icon": "verified",
                }
            )
        if "fundamentals_agent" in agent_analyses:
            consensus.append(
                {
                    "title": "Fundamental Strength",
                    "description": "Financial statements and quantitative metrics evaluated.",
                    "icon": "account_balance",
                }
            )
        return consensus
