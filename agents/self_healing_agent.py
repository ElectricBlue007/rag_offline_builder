"""
Optional self-healing agent for RAG pipeline outputs.

The deterministic pipeline should run first. This agent is an optional layer that
maps validation issues to exact source lines and, when requested, asks a local
LLM to produce a surgical Markdown repair.
"""

import json
import os
import subprocess
from typing import Any, Callable, Dict, List, Optional, Tuple


class SelfHealingAgent:
    def __init__(self, model_name: str = "qwen2.5:7b", llm_client: Optional[Callable[[str], str]] = None):
        self.model_name = model_name
        self.llm_client = llm_client
        self.max_context_window = 20

    def _call_model(self, prompt: str) -> str:
        """Call the configured LLM client."""
        if self.llm_client:
            return self.llm_client(prompt).strip()
        return self._call_ollama(prompt)

    def _call_ollama(self, prompt: str) -> str:
        """Call local Ollama model for reasoning and fixing."""
        try:
            result = subprocess.run(
                ["ollama", "run", self.model_name, prompt],
                capture_output=True,
                text=True,
                timeout=120,
                encoding="utf-8",
            )
            if result.returncode != 0:
                print(f"Model error: {result.stderr}")
                return ""
            return result.stdout.strip()
        except subprocess.TimeoutExpired:
            print("Model timeout. Skipping this fix.")
            return ""
        except Exception as exc:
            print(f"Connection error: {exc}")
            return ""

    def _extract_context(self, lines: List[str], line_idx: int) -> Tuple[str, int, int]:
        """Extract a window of lines around the issue."""
        start = max(0, line_idx - self.max_context_window // 2)
        end = min(len(lines), line_idx + self.max_context_window // 2)

        if line_idx < start:
            start = line_idx
        if line_idx >= end:
            end = line_idx + 1

        return "".join(lines[start:end]), start, end

    def _generate_fix_prompt(self, issue: Dict[str, Any], context: str, line_num: int) -> str:
        """Generate a strict prompt for the model."""
        issue_type = issue.get("issue_type", issue.get("type", "unknown"))
        rule_id = issue.get("rule_id", "unknown")
        description = issue.get("description", issue.get("rule_name", "Fix this issue"))

        return f"""
You are an expert Markdown Structural Engineer.
Your task is to fix a specific structural issue in a documentation file WITHOUT changing the original content words.

ISSUE DETAILS:
- Rule: {rule_id}
- Type: {issue_type}
- Description: {description}
- Target line: {line_num}

CONTEXT SNIPPET:
```markdown
{context}
```

INSTRUCTIONS:
1. Fix only the issue identified above.
2. Preserve technical terms, code, tables, links, and original meaning.
3. If scope is line-based, return only the corrected target line.
4. If the issue requires context, return only the minimal corrected Markdown snippet.
5. Output only Markdown. No explanations.

FIXED MARKDOWN:
"""

    def _apply_fix(
        self,
        original_lines: List[str],
        start_idx: int,
        end_idx: int,
        fixed_snippet: str,
    ) -> List[str]:
        """Replace the target line or context range with the fixed snippet."""
        if not fixed_snippet:
            return original_lines

        fixed_lines = fixed_snippet.splitlines(keepends=True)
        if fixed_lines and not fixed_lines[-1].endswith("\n"):
            fixed_lines[-1] += "\n"

        return original_lines[:start_idx] + fixed_lines + original_lines[end_idx:]

    def _extract_qualification_items(self, data: Dict[str, Any], mode: str = "issues") -> List[Dict[str, Any]]:
        """Read validation items from raw issue JSON or full process output."""
        qualification = data.get("phase_2_qualification", data)
        issues = qualification.get("issues", []) or []

        if mode in {"semantic-enhance", "warnings", "all"}:
            warnings = qualification.get("warnings", []) or []
            return issues + warnings

        return issues

    def _build_block_line_index(self, data: Dict[str, Any], content: str) -> Dict[str, int]:
        """Map block_id values from process output to one-based source line numbers."""
        blocks = data.get("phase_1_structure", {}).get("blocks", [])
        line_index: Dict[str, int] = {}
        search_from = 0

        for block in blocks:
            block_id = block.get("block_id")
            block_content = (block.get("content") or "").strip()
            if not block_id or not block_content:
                continue

            needle = block_content[: min(len(block_content), 120)].strip()
            pos = content.find(needle, search_from)
            if pos == -1:
                pos = content.find(needle)
            if pos == -1:
                continue

            line_index[block_id] = content.count("\n", 0, pos) + 1
            search_from = pos + len(needle)

        return line_index

    def _resolve_issue_line(
        self,
        issue: Dict[str, Any],
        block_line_index: Dict[str, int],
        content: str,
    ) -> Optional[int]:
        """Resolve a validation item to a one-based source line number when possible."""
        loc = issue.get("location")
        if isinstance(loc, dict) and loc.get("line"):
            return loc["line"]

        block_id = issue.get("block_id")
        if block_id in block_line_index:
            return block_line_index[block_id]

        context = issue.get("context") or {}
        for key in ("content", "heading"):
            value = context.get(key)
            if value:
                pos = content.find(str(value))
                if pos != -1:
                    return content.count("\n", 0, pos) + 1

        if isinstance(loc, str) and loc:
            pos = content.find(loc)
            if pos != -1:
                return content.count("\n", 0, pos) + 1

        return None

    def _target_range(self, line_num: int, context_start: int, context_end: int, scope: str) -> Tuple[int, int]:
        """Return zero-based replacement range for the requested scope."""
        if scope == "context":
            return context_start, context_end
        line_idx = line_num - 1
        return line_idx, line_idx + 1

    def collect_fixable_items(self, input_md: str, issues_json: str, mode: str = "issues") -> List[Dict[str, Any]]:
        """Return validation items that can be mapped to exact source lines."""
        with open(input_md, "r", encoding="utf-8") as file:
            content = file.read()
            lines = content.splitlines(keepends=True)

        with open(issues_json, "r", encoding="utf-8") as file:
            data = json.load(file)

        issues = self._extract_qualification_items(data, mode=mode)
        block_line_index = self._build_block_line_index(data, content)
        fixable_items = []

        for issue in issues:
            line_num = self._resolve_issue_line(issue, block_line_index, content)
            if not line_num:
                continue

            context, start_idx, end_idx = self._extract_context(lines, line_num - 1)
            fixable_items.append(
                {
                    "issue": issue,
                    "line": line_num,
                    "context": context,
                    "context_start": start_idx,
                    "context_end": end_idx,
                }
            )

        fixable_items.sort(key=lambda item: item["line"], reverse=True)
        return fixable_items

    def heal(
        self,
        input_md: str,
        issues_json: str,
        output_md: Optional[str] = None,
        mode: str = "issues",
        use_llm: bool = True,
        dry_run: bool = False,
        scope: str = "line",
    ) -> Optional[str]:
        """
        Heal a markdown file from validation output.

        dry_run or use_llm=False lists exact fixable lines without changing files.
        scope='line' replaces only the resolved target line.
        scope='context' replaces the extracted context window for complex issues.
        """
        if scope not in {"line", "context"}:
            raise ValueError("scope must be 'line' or 'context'")

        if not output_md:
            base, ext = os.path.splitext(input_md)
            output_md = f"{base}_fixed{ext}"

        print("Starting optional self-healing process...")
        print(f"Input: {input_md}")
        print(f"Issues: {issues_json}")

        fixable_items = self.collect_fixable_items(input_md, issues_json, mode=mode)
        if not fixable_items:
            print("No fixable validation items found for this mode.")
            return None

        print(f"Found {len(fixable_items)} fixable validation item(s).")
        for item in reversed(fixable_items):
            issue = item["issue"]
            label = issue.get("description", issue.get("rule_name", "validation item"))
            print(f"   line {item['line']}: Rule {issue.get('rule_id')} - {label}")

        if dry_run or not use_llm:
            print("Dry run complete. No file was changed.")
            return None

        with open(input_md, "r", encoding="utf-8") as file:
            lines = file.read().splitlines(keepends=True)

        fixed_count = 0
        for index, item in enumerate(fixable_items, start=1):
            issue = item["issue"]
            line_num = item["line"]
            print(f"\n[{index}/{len(fixable_items)}] Fixing Rule {issue.get('rule_id')} at line {line_num}...")

            prompt = self._generate_fix_prompt(issue, item["context"], line_num)
            fixed_snippet = self._call_model(prompt)

            if fixed_snippet:
                start_idx, end_idx = self._target_range(
                    line_num,
                    item["context_start"],
                    item["context_end"],
                    scope,
                )
                lines = self._apply_fix(lines, start_idx, end_idx, fixed_snippet)
                print("   Applied fix.")
                fixed_count += 1
            else:
                print("   Failed to generate fix. Skipping.")

        with open(output_md, "w", encoding="utf-8") as file:
            file.writelines(lines)

        print("\nHealing complete.")
        print(f"   Fixed: {fixed_count}/{len(fixable_items)} validation item(s)")
        print(f"   Output saved to: {output_md}")
        return output_md


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 3:
        print("Usage: python -m agents.self_healing_agent <input.md> <issues.json> [output.md]")
        sys.exit(1)

    agent = SelfHealingAgent()
    agent.heal(sys.argv[1], sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else None)
