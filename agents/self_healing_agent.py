"""
Self-Healing Agent for RAG Pipeline
Automatically fixes structural Markdown issues using local LLMs without rewriting content.
"""

import json
import os
import re
import subprocess
from typing import List, Dict, Any, Optional
from pathlib import Path

class SelfHealingAgent:
    def __init__(self, model_name: str = "qwen2.5:7b"):
        self.model_name = model_name
        self.max_context_window = 20  # Lines to show around the error
        
    def _call_ollama(self, prompt: str) -> str:
        """Call local Ollama model for reasoning and fixing."""
        try:
            result = subprocess.run(
                ["ollama", "run", self.model_name, prompt],
                capture_output=True,
                text=True,
                timeout=120,  # 2 minutes per fix
                encoding='utf-8'
            )
            if result.returncode != 0:
                print(f"⚠️ Model error: {result.stderr}")
                return ""
            return result.stdout.strip()
        except subprocess.TimeoutExpired:
            print("⚠️ Model timeout. Skipping this fix.")
            return ""
        except Exception as e:
            print(f"⚠️ Connection error: {e}")
            return ""

    def _extract_context(self, lines: List[str], line_idx: int) -> tuple[str, int, int]:
        """Extract a window of lines around the issue."""
        start = max(0, line_idx - self.max_context_window // 2)
        end = min(len(lines), line_idx + self.max_context_window // 2)
        
        # Ensure we include the actual broken line
        if line_idx < start: start = line_idx
        if line_idx >= end: end = line_idx + 1
        
        context = "".join(lines[start:end])
        return context, start, end

    def _generate_fix_prompt(self, issue: Dict[str, Any], context: str, line_num: int) -> str:
        """Generate a strict prompt for the model."""
        issue_type = issue.get('type', 'unknown')
        rule_id = issue.get('rule_id', 'unknown')
        description = issue.get('description', 'Fix this issue')
        
        return f"""
You are an expert Markdown Structural Engineer. 
Your task is to fix a specific structural issue in a documentation file WITHOUT changing the original content words.

ISSUE DETAILS:
- Rule: {rule_id}
- Type: {issue_type}
- Description: {description}
- Location: Around line {line_num} in the provided context.

CONTEXT SNIPPET (Lines {line_num - 10} to {line_num + 10}):
```markdown
{context}
```

INSTRUCTIONS:
1. Identify the exact broken part in the context.
2. Generate ONLY the corrected version of that specific part.
3. DO NOT summarize, rewrite, or change technical terms/code.
4. Preserve all original whitespace and indentation unless it causes the error.
5. If the issue is a missing heading, insert the most logical heading based on the content.
6. If the issue is a broken code fence, close it properly.
7. Output ONLY the fixed markdown snippet. No explanations. No conversational text.

FIXED SNIPPET:
"""

    def _apply_fix(self, original_lines: List[str], start_idx: int, end_idx: int, fixed_snippet: str) -> List[str]:
        """Surgically replace the broken lines with the fixed snippet."""
        if not fixed_snippet:
            return original_lines
            
        fixed_lines = fixed_snippet.splitlines(keepends=True)
        # Ensure newlines are preserved correctly
        if not fixed_lines[-1].endswith('\n'):
            fixed_lines[-1] += '\n'
            
        new_lines = original_lines[:start_idx] + fixed_lines + original_lines[end_idx:]
        return new_lines

    def heal(self, input_md: str, issues_json: str, output_md: Optional[str] = None):
        """
        Main healing process.
        1. Load issues.
        2. Sort by line number (bottom-up to preserve indices).
        3. Fix each issue surgically.
        """
        if not output_md:
            base, ext = os.path.splitext(input_md)
            output_md = f"{base}_fixed{ext}"
            
        print(f"🩺 Starting Self-Healing Process...")
        print(f"📄 Input: {input_md}")
        print(f"📋 Issues: {issues_json}")
        
        # Load content
        with open(input_md, 'r', encoding='utf-8') as f:
            content = f.read()
            lines = content.splitlines(keepends=True)
            
        # Load issues
        with open(issues_json, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        issues = data.get('issues', [])
        if not issues:
            print("✅ No issues found. Document is clean.")
            return
            
        print(f"🔍 Found {len(issues)} issues to fix.")
        
        # Sort issues by line number descending (fix bottom-up to avoid index shifting)
        # We assume issues have a 'line' or 'location' field. If not, we skip.
        fixable_issues = []
        for issue in issues:
            loc = issue.get('location', {})
            line_num = loc.get('line')
            if line_num:
                issue['_original_line'] = line_num
                fixable_issues.append(issue)
            else:
                print(f"⚠️ Skipping issue without line number: {issue.get('type')}")
                
        fixable_issues.sort(key=lambda x: x['_original_line'], reverse=True)
        
        fixed_count = 0
        for i, issue in enumerate(fixable_issues):
            line_num = issue['_original_line']
            # Adjust for 0-based index
            line_idx = line_num - 1
            
            print(f"\n[{i+1}/{len(fixable_issues)}] Fixing Rule {issue.get('rule_id')} at line {line_num}...")
            
            # Extract context
            context, start_idx, end_idx = self._extract_context(lines, line_idx)
            
            # Generate prompt
            prompt = self._generate_fix_prompt(issue, context, line_num)
            
            # Call Model
            print(f"   🤖 Consulting {self.model_name}...")
            fixed_snippet = self._call_ollama(prompt)
            
            if fixed_snippet:
                # Apply fix
                lines = self._apply_fix(lines, start_idx, end_idx, fixed_snippet)
                print(f"   ✅ Applied fix.")
                fixed_count += 1
            else:
                print(f"   ❌ Failed to generate fix. Skipping.")
                
        # Write output
        with open(output_md, 'w', encoding='utf-8') as f:
            f.writelines(lines)
            
        print(f"\n🎉 Healing Complete!")
        print(f"   Fixed: {fixed_count}/{len(fixable_issues)} issues")
        print(f"   Output saved to: {output_md}")
        return output_md

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 3:
        print("Usage: python -m agents.self_healing_agent <input.md> <issues.json> [output.md]")
        sys.exit(1)
        
    agent = SelfHealingAgent()
    agent.heal(sys.argv[1], sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else None)
