"""Optional LLM-backed chunk repair pass.

The repair agent consumes the automated chunk review report, asks an LLM to
rewrite only chunks that need repair/rechunking, replaces the chunk text in the
index, and then runs the review agent again against the repaired chunks.
"""

from __future__ import annotations

import copy
import hashlib
import json
import re
import subprocess
import unicodedata
from dataclasses import dataclass
from typing import Any, Callable, Iterable

from agents.chunk_review_agent import ChunkReviewAgent
from agents.rule_repair_prompt_agent import RuleRepairPromptAgent


LLMClient = Callable[[str], str]


@dataclass
class ChunkRepairAttempt:
    chunk_id: str
    before_score: int
    after_score: int | None
    before_decision: str
    after_decision: str | None
    issue_rule_ids: list[int]
    status: str
    model: str
    original_sha256: str
    repaired_sha256: str | None = None
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "before_score": self.before_score,
            "after_score": self.after_score,
            "before_decision": self.before_decision,
            "after_decision": self.after_decision,
            "issue_rule_ids": self.issue_rule_ids,
            "status": self.status,
            "model": self.model,
            "original_sha256": self.original_sha256,
            "repaired_sha256": self.repaired_sha256,
            "error": self.error,
        }


class ChunkRepairAgent:
    """Repair reviewed chunks with an LLM and validate the repaired index."""

    DEFAULT_REPAIR_DECISIONS = ("REPAIR", "RECHUNK")

    def __init__(
        self,
        review_agent: ChunkReviewAgent | None = None,
        model_name: str = "qwen2.5:7b",
        llm_client: LLMClient | None = None,
        timeout_seconds: int = 120,
        max_attempts: int = 2,
        prompt_agent: RuleRepairPromptAgent | None = None,
    ) -> None:
        self.review_agent = review_agent or ChunkReviewAgent()
        self.model_name = model_name
        self.llm_client = llm_client
        self.timeout_seconds = timeout_seconds
        self.max_attempts = max(1, max_attempts)
        self.prompt_agent = prompt_agent or RuleRepairPromptAgent()

    def repair_index(
        self,
        index: dict[str, Any],
        embedding_model: str = "e5",
        repair_decisions: Iterable[str] | None = None,
    ) -> dict[str, Any]:
        repaired_index = copy.deepcopy(index)
        decisions = set(repair_decisions or self.DEFAULT_REPAIR_DECISIONS)

        initial_report = repaired_index.get("chunk_review") or self.review_agent.review_index(
            repaired_index,
            embedding_model=embedding_model,
        ).to_dict()
        repaired_index["chunk_review"] = initial_report

        reviews_by_id = {review["chunk_id"]: review for review in initial_report.get("chunk_reviews", [])}
        chunks = repaired_index.get("flat_chunk_view", [])
        attempts: list[ChunkRepairAttempt] = []

        for chunk in chunks:
            chunk_id = chunk.get("chunk_id")
            review = reviews_by_id.get(chunk_id)
            if not review:
                continue
            if review.get("decision") not in decisions:
                continue

            attempt = self._repair_chunk(chunk, review, embedding_model=embedding_model)
            attempts.append(attempt)

        final_report = self.review_agent.review_index(repaired_index, embedding_model=embedding_model).to_dict()
        final_by_id = {review["chunk_id"]: review for review in final_report.get("chunk_reviews", [])}

        for attempt in attempts:
            final_review = final_by_id.get(attempt.chunk_id)
            if final_review:
                attempt.after_score = final_review.get("final_score")
                attempt.after_decision = final_review.get("decision")

        repaired_index["chunk_review"] = final_report
        repaired_index["chunk_repair_report"] = self._build_report(
            attempts=attempts,
            before_summary=initial_report.get("summary", {}),
            after_summary=final_report.get("summary", {}),
            repair_decisions=sorted(decisions),
            embedding_model=embedding_model,
        )
        return repaired_index

    def _repair_chunk(self, chunk: dict[str, Any], review: dict[str, Any], embedding_model: str) -> ChunkRepairAttempt:
        original_text = chunk.get("text", "") or ""
        attempt = ChunkRepairAttempt(
            chunk_id=chunk.get("chunk_id", ""),
            before_score=int(review.get("final_score", 0)),
            after_score=None,
            before_decision=review.get("decision", "UNKNOWN"),
            after_decision=None,
            issue_rule_ids=[issue.get("rule_id") for issue in review.get("issues", []) if issue.get("action") != "NONE"],
            status="SKIPPED",
            model=self.model_name,
            original_sha256=self._sha256(original_text),
        )

        try:
            repaired_text, validation_errors = self._generate_valid_repair(
                chunk,
                review,
                embedding_model=embedding_model,
            )
            if not repaired_text.strip():
                attempt.status = "FAILED"
                attempt.error = validation_errors[-1] if validation_errors else "Model returned empty repaired text."
                return attempt

            chunk["text"] = repaired_text
            chunk["token_estimate"] = self._estimate_tokens(repaired_text)
            chunk.setdefault("debug_metadata", {})
            chunk["debug_metadata"]["chunk_repair"] = {
                "model": self.model_name,
                "before_decision": attempt.before_decision,
                "before_score": attempt.before_score,
                "issue_rule_ids": attempt.issue_rule_ids,
                "original_sha256": attempt.original_sha256,
            }
            attempt.status = "REPAIRED"
            attempt.repaired_sha256 = self._sha256(repaired_text)
            return attempt
        except Exception as error:  # pragma: no cover - exercised through failure behavior, not exact type.
            attempt.status = "FAILED"
            attempt.error = str(error)
            return attempt

    def _build_prompt(
        self,
        chunk: dict[str, Any],
        review: dict[str, Any],
        embedding_model: str,
        feedback: str = "",
    ) -> str:
        return self.prompt_agent.chunk_prompt(
            chunk=chunk,
            review=review,
            embedding_model=embedding_model,
            feedback=feedback,
        )

    def _call_model(self, prompt: str) -> str:
        if self.llm_client:
            return self.llm_client(prompt)

        result = subprocess.run(
            ["ollama", "run", self.model_name, prompt],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=self.timeout_seconds,
            check=False,
        )
        if result.returncode != 0:
            raise RuntimeError(result.stderr.strip() or f"Ollama exited with code {result.returncode}.")
        return result.stdout

    @staticmethod
    def _sanitize_model_output(output: str) -> str:
        text = output.strip()
        fenced = re.match(r"^```(?:markdown|md|text)?\s*(.*?)\s*```$", text, re.DOTALL | re.IGNORECASE)
        if fenced:
            text = fenced.group(1).strip()
        text = re.sub(r"^\s*(?:fixed chunk|repaired chunk|fixed markdown|replacement)\s*:\s*", "", text, flags=re.IGNORECASE)
        return text

    def _generate_valid_repair(
        self,
        chunk: dict[str, Any],
        review: dict[str, Any],
        embedding_model: str,
    ) -> tuple[str, list[str]]:
        feedback = ""
        errors: list[str] = []
        original_text = chunk.get("text", "") or ""
        for _ in range(self.max_attempts):
            prompt = self._build_prompt(chunk, review, embedding_model=embedding_model, feedback=feedback)
            repaired_text = self._sanitize_model_output(self._call_model(prompt))
            repaired_text = self._postprocess_text(repaired_text, chunk, embedding_model=embedding_model)
            valid, error = self._validate_repair(original_text, repaired_text)
            if valid:
                return repaired_text, errors
            errors.append(error)
            feedback = error
        return "", errors

    @staticmethod
    def _validate_repair(original_text: str, repaired_text: str) -> tuple[bool, str]:
        if not repaired_text.strip():
            return False, "Model returned empty repaired text."
        if re.search(r"\b(here is|i fixed|the issue|explanation|note:)\b", repaired_text, re.IGNORECASE):
            return False, "Model returned explanation text instead of only chunk text."
        if repaired_text.count("```") % 2:
            return False, "Repaired chunk has unbalanced code fences."
        original_lower = original_text.lower()
        repaired_lower = repaired_text.lower()
        if original_lower.count("<table") == original_lower.count("</table"):
            if repaired_lower.count("<table") != repaired_lower.count("</table"):
                return False, "Repaired chunk has an unbalanced HTML table."
        if len(repaired_text) > max(len(original_text) * 5, len(original_text) + 1600):
            return False, "Repaired chunk expanded too much."
        return True, ""

    @staticmethod
    def _postprocess_text(text: str, chunk: dict[str, Any], embedding_model: str) -> str:
        normalized = unicodedata.normalize("NFC", text).strip()
        heading_path = " > ".join(chunk.get("heading_path") or [])
        if heading_path and heading_path.lower() not in normalized.lower()[: max(160, len(heading_path) + 40)]:
            normalized = f"{heading_path}\n\n{normalized}"
        if embedding_model.lower().startswith("e5") and not normalized.lower().startswith("passage:"):
            normalized = f"passage: {normalized}"
        return normalized

    @staticmethod
    def _estimate_tokens(text: str) -> int:
        return max(len(re.findall(r"\S+", text)), (len(text) + 3) // 4)

    @staticmethod
    def _sha256(text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    @staticmethod
    def _build_report(
        attempts: list[ChunkRepairAttempt],
        before_summary: dict[str, Any],
        after_summary: dict[str, Any],
        repair_decisions: list[str],
        embedding_model: str,
    ) -> dict[str, Any]:
        repaired = [attempt for attempt in attempts if attempt.status == "REPAIRED"]
        failed = [attempt for attempt in attempts if attempt.status == "FAILED"]
        return {
            "agent": "ChunkRepairAgent",
            "enabled": True,
            "model": attempts[0].model if attempts else None,
            "embedding_model": embedding_model,
            "repair_decisions": repair_decisions,
            "attempted_count": len(attempts),
            "repaired_count": len(repaired),
            "failed_count": len(failed),
            "before_summary": before_summary,
            "after_summary": after_summary,
            "ready_for_embedding": after_summary.get("ready_for_embedding", False),
            "repairs": [attempt.to_dict() for attempt in attempts],
        }
