#!/usr/bin/env python3
"""
doctor.py -- the preflight check: "is this machine ready, and is my data sane?"

Why this exists
---------------
This project has many moving parts (Ollama models, a vector database, a
keyword index, several config sections). When something is wrong, the error
usually appears far away from its cause: a missing model shows up as an empty
answer, a stale index shows up as a wrong citation.

`doctor` checks every precondition ONCE, in order, and prints a plain-language
verdict plus the exact next command to run. It changes nothing: no writes, no
downloads, no model loading.

Run it:
    python rag_pipeline/run_doctor.py
    python rag_pipeline/run_doctor.py --out reports        # also write Markdown

Design note: this module reads settings.yaml as RAW YAML on purpose. It must
be able to tell you "your settings.yaml is broken" without depending on the
typed config layer that would itself fail on a broken file.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import platform
import sys
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent))

from rag_pipeline.observability import i18n
from rag_pipeline.observability.i18n import Bi

OK, WARN, FAIL, SKIP = "\u2705", "\u26a0\ufe0f", "\u274c", "\u23ed"

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SETTINGS = PROJECT_ROOT / "settings.yaml"


@dataclass
class Check:
    group: Bi
    name: Bi
    status: str
    detail: Bi
    fix: Optional[Bi] = None

    def to_dict(self) -> dict:
        return {"group": self.group[0], "name": self.name[0], "status": self.status,
                "detail": self.detail[0], "fix": (self.fix or ("", ""))[0]}


@dataclass
class Doctor:
    settings_path: Path = DEFAULT_SETTINGS
    checks: List[Check] = field(default_factory=list)
    raw: Dict[str, Any] = field(default_factory=dict)

    # -- helpers ----------------------------------------------------------

    def add(self, group: Bi, name: Bi, status: str, detail: Bi, fix: Optional[Bi] = None) -> None:
        self.checks.append(Check(group, name, status, detail, fix))

    def get(self, dotted: str, default=None):
        """Read a dotted key out of the raw YAML, never raising."""
        node: Any = self.raw
        for part in dotted.split("."):
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return node

    def _path(self, dotted: str, default: str) -> Path:
        """Resolve a settings path the same way the CLIs do: relative to CWD."""
        return Path(str(self.get(dotted, default))).expanduser()

    # -- 1. environment ---------------------------------------------------

    def check_python(self) -> None:
        g = ("Environment", "محیط اجرا")
        v = sys.version_info
        ok = v >= (3, 10)
        self.add(g, ("Python version", "نسخه پایتون"), OK if ok else FAIL,
                 (f"{platform.python_version()} on {platform.system()}",
                  f"{platform.python_version()} روی {platform.system()}"),
                 None if ok else ("This project uses modern type syntax; use Python 3.10 or newer.",
                                  "این پروژه به پایتون 3.10 یا بالاتر نیاز دارد."))
        self.add(g, ("Working directory", "پوشه جاری"), OK,
                 (f"{Path.cwd()} - all ./ paths in settings.yaml resolve from here",
                  f"{Path.cwd()} - همه مسیرهای ./ در settings.yaml از اینجا حساب می‌شوند"))

    def check_settings(self) -> None:
        g = ("Configuration", "تنطیمات")
        name = ("settings.yaml", "settings.yaml")
        if not self.settings_path.exists():
            self.add(g, name, FAIL,
                     (f"Not found at {self.settings_path}", f"در {self.settings_path} پیدا نشد"),
                     ("Without it every value falls back to the defaults in config.py, which may not match your models.",
                      "بدون آن همه مقادیر به پیش‌فرض‌های config.py برمی‌گردند."))
            return
        try:
            import yaml
        except ImportError:
            self.add(g, name, FAIL, ("pyyaml is not installed, so settings.yaml cannot be read.",
                                     "پکیج pyyaml نصب نیست پس settings.yaml خوانده نمی‌شود."),
                     ("pip install pyyaml", "pip install pyyaml"))
            return
        try:
            self.raw = yaml.safe_load(self.settings_path.read_text(encoding="utf-8")) or {}
            top = ", ".join(sorted(self.raw.keys()))
            self.add(g, name, OK,
                     (f"Parsed {self.settings_path} - sections: {top}",
                      f"خوانده شد: {self.settings_path}"))
        except Exception as exc:
            self.add(g, name, FAIL,
                     (f"Found but could NOT be parsed: {exc}", f"پیدا شد ولی خوانده نشد: {exc}"),
                     ("Fix the YAML indentation at the line named in the error.",
                      "تورفتگی YAML را در خط گفته‌شده اصلاح کنید."))

    def check_dependencies(self) -> None:
        g = ("Dependencies", "پکیج‌ها")
        required = [
            ("yaml", "pyyaml", ("read settings.yaml", "خواندن settings.yaml")),
            ("bs4", "beautifulsoup4", ("turn HTML tables into sentences", "تبدیل جدول HTML به متن")),
            ("lxml", "lxml", ("parse those tables quickly", "تحلیل سریع جداول")),
            ("ollama", "ollama", ("talk to the local models", "ارتباط با مدل‌های محلی")),
            ("chromadb", "chromadb", ("store and search vectors", "ذخیره و جستجوی وکتور")),
            ("langchain_experimental", "langchain-experimental", ("split by meaning", "تقسیم معنایی")),
            ("langchain_community", "langchain-community", ("embeddings adapter used by the splitter", "واسط تعبیه برای تقسیم‌کننده")),
            ("flask", "flask", ("serve the chat web UI", "اجرای رابط وب گفتگو")),
        ]
        optional = [
            ("sentence_transformers", "sentence-transformers",
             ("only for reranker backend: huggingface", "فقط برای بک‌اند huggingface ریرنکر")),
            ("transformers", "transformers",
             ("only for reranker backend: huggingface", "فقط برای بک‌اند huggingface ریرنکر")),
            ("ragas", "ragas", ("only for run_evaluation.py", "فقط برای run_evaluation.py")),
            ("rank_bm25", "rank-bm25",
             ("optional: a pure-Python BM25 fallback ships with the project",
              "اختیاری: نسخه جایگزین BM25 داخل پروژه هست")),
        ]
        missing_required = []
        for mod, pkg, why in required:
            present = importlib.util.find_spec(mod) is not None
            if not present:
                missing_required.append(pkg)
            self.add(g, (pkg, pkg), OK if present else FAIL,
                     (f"{'installed' if present else 'MISSING'} - needed to {why[0]}",
                      f"{'نصب است' if present else 'نصب نیست'} - لازم برای: {why[1]}"),
                     None if present else (f"pip install {pkg}", f"pip install {pkg}"))
        for mod, pkg, why in optional:
            present = importlib.util.find_spec(mod) is not None
            self.add(g, (pkg, pkg), OK if present else SKIP,
                     (f"{'installed' if present else 'not installed'} - {why[0]}",
                      f"{'نصب است' if present else 'نصب نیست'} - {why[1]}"))
        if missing_required:
            self.add(g, ("Install command", "دستور نصب"), WARN,
                     ("pip install -r rag_pipeline/requirements.txt",
                      "pip install -r rag_pipeline/requirements.txt"),
                     (f"Missing: {', '.join(missing_required)}", f"ناقص: {', '.join(missing_required)}"))

    # -- 2. models --------------------------------------------------------

    def check_ollama(self) -> None:
        g = ("Models (Ollama)", "مدل‌ها (Ollama)")
        host = str(self.get("ollama.host", "http://localhost:11434")).rstrip("/")
        url = f"{host}/api/tags"
        try:
            with urllib.request.urlopen(url, timeout=5) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
        except Exception as exc:
            self.add(g, ("Ollama server", "سرور Ollama"), FAIL,
                     (f"Not reachable at {host} ({exc})", f"در {host} در دسترس نیست ({exc})"),
                     ("Start it with `ollama serve`, then re-run doctor. Chunking, embedding, answering and evaluation all need it.",
                      "با دستور ollama serve روشنش کنید و دوباره doctor را اجرا کنید."))
            self._report_configured_models(g, installed=None)
            return

        installed = sorted({m.get("name", "") for m in payload.get("models", [])})
        self.add(g, ("Ollama server", "سرور Ollama"), OK,
                 (f"Reachable at {host} with {len(installed)} model(s) installed",
                  f"در {host} در دسترس است با {len(installed)} مدل نصب‌شده"))
        self._report_configured_models(g, installed=installed)

    def _report_configured_models(self, g: Bi, installed: Optional[List[str]]) -> None:
        """
        Compare every model tag in the `models:` block against what Ollama has.

        Heuristic, stated openly: entries containing ":" are treated as Ollama
        tags; HuggingFace ids such as Qwen/Qwen3-Reranker-0.6B are not Ollama
        models and are reported as informational only.
        """
        models = self.get("models", {}) or {}
        if not isinstance(models, dict) or not models:
            self.add(g, ("Configured models", "مدل‌های تنطیم‌شده"), WARN,
                     ("No `models:` block in settings.yaml; defaults from config.py will be used.",
                      "بخش models در settings.yaml نیست؛ پیش‌فرض‌های config.py استفاده می‌شوند."))
            return
        for role, tag in models.items():
            tag = str(tag)
            if "/" in tag and ":" not in tag:
                self.add(g, (f"models.{role}", f"models.{role}"), SKIP,
                         (f"{tag} - a HuggingFace id, downloaded by the ML libraries, not by Ollama",
                          f"{tag} - شناسه HuggingFace است و توسط Ollama دانلود نمی‌شود"))
                continue
            if installed is None:
                self.add(g, (f"models.{role}", f"models.{role}"), SKIP,
                         (f"{tag} - cannot verify while Ollama is unreachable",
                          f"{tag} - تا زمانی که Ollama در دسترس نیست قابل بررسی نیست"))
                continue
            present = tag in installed or f"{tag}:latest" in installed
            self.add(g, (f"models.{role}", f"models.{role}"), OK if present else FAIL,
                     (f"{tag} - {'installed' if present else 'NOT installed'}",
                      f"{tag} - {'نصب است' if present else 'نصب نیست'}"),
                     None if present else (f"ollama pull {tag}", f"ollama pull {tag}"))

    # -- 3. data ----------------------------------------------------------

    def check_data(self) -> Tuple[int, int, Optional[int], Optional[int]]:
        g = ("Your data", "داده‌های شما")
        input_dir = self._path("paths.input_dir", "./outputs")
        chunks_dir = self._path("paths.chunks_dir", "./chunks")
        chroma_dir = self._path("paths.chroma_dir", "./chroma_db")
        bm25_path = self._path("paths.bm25_index_path", "./bm25_index.pkl")
        glob = str(self.get("chunking.md_glob", "**/*.md"))

        md_count = len(list(input_dir.glob(glob))) if input_dir.exists() else 0
        self.add(g, ("Input .md files", "فایل‌های ورودی md"),
                 OK if md_count else FAIL,
                 (f"{md_count} file(s) matching {glob} in {input_dir}",
                  f"{md_count} فایل مطابق {glob} در {input_dir}"),
                 None if md_count else ("Put your OCR markdown files in that folder (paths.input_dir), or point it somewhere else.",
                                        "فایل‌های مارک‌داون OCR را در همان پوشه بگذارید."))

        chunk_files = list(chunks_dir.rglob("*.chunks.json")) if chunks_dir.exists() else []
        chunk_total = 0
        for p in chunk_files:
            try:
                chunk_total += int(json.loads(p.read_text(encoding="utf-8")).get("total_chunks", 0))
            except Exception:
                pass
        self.add(g, ("Chunks on disk", "قطعات روی دیسک"),
                 OK if chunk_files else WARN,
                 (f"{chunk_total} chunk(s) in {len(chunk_files)} file(s) under {chunks_dir}",
                  f"{chunk_total} قطعه در {len(chunk_files)} فایل در {chunks_dir}"),
                 None if chunk_files else ("Run: python rag_pipeline/run_pipeline.py",
                                           "اجرا کنید: python rag_pipeline/run_pipeline.py"))

        chroma_count = self._chroma_count(chroma_dir)
        if not chroma_dir.exists():
            chroma_detail = (f"{chroma_dir} does not exist yet", f"پوشه {chroma_dir} هنوز ساخته نشده")
        elif chroma_count is None:
            chroma_detail = (f"{chroma_dir} exists but could not be read here",
                             f"پوشه {chroma_dir} هست ولی خوانده نشد")
        else:
            chroma_detail = (f"{chroma_count} record(s) in {chroma_dir}",
                             f"{chroma_count} رکورد در {chroma_dir}")
        self.add(g, ("Vector database (Chroma)", "پایگاه وکتور (Chroma)"),
                 OK if chroma_count else WARN, chroma_detail,
                 None if chroma_count else ("Run: python rag_pipeline/run_ingestion.py",
                                            "اجرا کنید: python rag_pipeline/run_ingestion.py"))

        bm25_size = self._bm25_size(bm25_path)
        self.add(g, ("Keyword index (BM25)", "ایندکس کلیدواژه (BM25)"),
                 OK if bm25_size else WARN,
                 (f"{bm25_path}: " + (f"{bm25_size} document(s) indexed" if bm25_size is not None
                                      else "missing or unreadable"),
                  f"{bm25_path}: " + (f"{bm25_size} سند ایندکس‌شده" if bm25_size is not None else "نیست یا خوانده نشد")),
                 None if bm25_size else ("Run: python rag_pipeline/run_bm25_index.py",
                                         "اجرا کنید: python rag_pipeline/run_bm25_index.py"))
        return md_count, chunk_total, chroma_count, bm25_size

    def _chroma_count(self, chroma_dir: Path) -> Optional[int]:
        if not chroma_dir.exists() or importlib.util.find_spec("chromadb") is None:
            return None
        try:
            import chromadb
            client = chromadb.PersistentClient(path=str(chroma_dir))
            name = str(self.get("paths.collection_name", "rag_docs"))
            return client.get_or_create_collection(name=name).count()
        except Exception:
            return None

    def _bm25_size(self, bm25_path: Path) -> Optional[int]:
        if not bm25_path.exists():
            return None
        try:
            import pickle
            with bm25_path.open("rb") as fh:
                state = pickle.load(fh)
            return len(state.get("chunk_ids", []))
        except Exception:
            return None

    def check_drift(self, chunk_total: int, chroma_count: Optional[int], bm25_size: Optional[int]) -> None:
        """
        The most common silent failure in this project: the vector store is
        upsert-only while the BM25 index is rebuilt from scratch, so deleting
        a document leaves ghosts in Chroma but not in BM25.
        """
        g = ("Index consistency", "هماهنگی ایندکس‌ها")
        if chroma_count is None or bm25_size is None:
            self.add(g, ("Chroma vs BM25", "Chroma در برابر BM25"), SKIP,
                     ("Cannot compare: one of the two indexes could not be read.",
                      "قابل مقایسه نیست: یکی از دو ایندکس خوانده نشد."))
            return
        if chroma_count == bm25_size == chunk_total:
            self.add(g, ("Chroma vs BM25 vs disk", "Chroma، BM25، دیسک"), OK,
                     (f"All three agree: {chunk_total} chunks.", f"هر سه یکسانند: {chunk_total} قطعه."))
            return
        self.add(g, ("Chroma vs BM25 vs disk", "Chroma، BM25، دیسک"), WARN,
                 (f"Chroma has {chroma_count}, BM25 has {bm25_size}, disk has {chunk_total}. "
                  "Chroma only ever adds/overwrites, so a number higher than disk means deleted "
                  "documents are still being returned by vector search.",
                  f"Chroma {chroma_count}، BM25 {bm25_size}، دیسک {chunk_total}. Chroma فقط اضافه/بازنویسی می‌کند، "
                  "پس عدد بیشتر از دیسک یعنی اسناد حذف‌شده هنوز در نتایج می‌آیند."),
                 ("To resync from scratch: delete the folder in paths.chroma_dir and the file in "
                  "paths.bm25_index_path, then run run_ingestion.py and run_bm25_index.py again.",
                  "برای هماهنگ‌سازی از صفر: پوشه chroma_dir و فایل bm25_index_path را حذف کنید و دو بارگذاری را دوباره اجرا کنید."))

    # -- 4. config sanity -------------------------------------------------

    def check_config_sanity(self) -> None:
        g = ("Config sanity", "سلامت تنطیمات")

        gen_ctx = self.get("generation.num_ctx")
        ollama_ctx = self.get("ollama.num_ctx")
        if isinstance(gen_ctx, int) and isinstance(ollama_ctx, int) and gen_ctx > ollama_ctx:
            self.add(g, ("Context window", "پنجره زمینه"), WARN,
                     (f"generation.num_ctx ({gen_ctx}) is larger than ollama.num_ctx ({ollama_ctx}). "
                      "The generation step wins, so it will ask the model for a bigger window than the "
                      "project-wide default - fine if your GPU can hold it, an out-of-memory error if not.",
                      f"مقدار generation.num_ctx ({gen_ctx}) از ollama.num_ctx ({ollama_ctx}) بیشتر است. "
                      "اگر حافطه GPU کافی نباشد خطای کمبود حافطه می‌دهد."),
                     ("Lower generation.num_ctx if the model fails to load.",
                      "اگر مدل لود نشد generation.num_ctx را کم کنید."))
        else:
            self.add(g, ("Context window", "پنجره زمینه"), OK,
                     (f"generation.num_ctx={gen_ctx}, ollama.num_ctx={ollama_ctx}",
                      f"generation.num_ctx={gen_ctx}، ollama.num_ctx={ollama_ctx}"))

        backend = str(self.get("retrieval.reranker.backend", "huggingface"))
        rerank_on = bool(self.get("retrieval.reranker.enabled", False))
        if backend == "huggingface" and importlib.util.find_spec("sentence_transformers") is None:
            self.add(g, ("Reranker backend", "بک‌اند ریرنکر"), FAIL if rerank_on else WARN,
                     ("backend is 'huggingface' but sentence-transformers is not installed.",
                      "بک‌اند huggingface است ولی sentence-transformers نصب نیست."),
                     ("Either set retrieval.reranker.backend: ollama (no extra install), or run "
                      "pip install -r rag_pipeline/requirements-rerank.txt",
                      "یا backend را ollama کنید، یا requirements-rerank.txt را نصب کنید."))
        else:
            self.add(g, ("Reranker backend", "بک‌اند ریرنکر"), OK,
                     (f"backend={backend}, enabled by default={rerank_on}",
                      f"backend={backend}، پیش‌فرض فعال={rerank_on}"))

        n_results = self.get("generation.n_results")
        max_chunks = self.get("generation.context_builder.max_chunks",
                              self.get("context_builder.max_chunks"))
        if isinstance(n_results, int) and isinstance(max_chunks, int) and n_results > max_chunks:
            self.add(g, ("Retrieved vs used", "بازیابی‌شده در برابر استفاده‌شده"), WARN,
                     (f"Search returns {n_results} chunks but only {max_chunks} are put in the prompt; "
                      "the rest are fetched and thrown away.",
                      f"جستجو {n_results} قطعه می‌دهد ولی فقط {max_chunks} در پرامپت می‌رود."),
                     ("Make context_builder.max_chunks equal to generation.n_results, or lower n_results.",
                      "max_chunks را با n_results برابر کنید یا n_results را کم کنید."))
        else:
            self.add(g, ("Retrieved vs used", "بازیابی‌شده در برابر استفاده‌شده"), OK,
                     (f"generation.n_results={n_results}, context_builder.max_chunks={max_chunks}",
                      f"generation.n_results={n_results}، context_builder.max_chunks={max_chunks}"))

        if self.get("retrieval.reranker.final_n_results") is not None:
            self.add(g, ("Dead setting", "تنطیم بی‌استفاده"), WARN,
                     ("retrieval.reranker.final_n_results is never used: every CLI overrides it with "
                      "--n-results at call time. Changing it has no effect.",
                      "مقدار retrieval.reranker.final_n_results استفاده نمی‌شود؛ همه CLI‌ها با --n-results آن را بازنویسی می‌کنند."),
                     ("Use --n-results on the command line instead.",
                      "از --n-results در خط فرمان استفاده کنید."))

        qpath = self.get("evaluation.questions_path")
        if qpath and not Path(str(qpath)).exists():
            self.add(g, ("Evaluation questions", "سوالات ارزیابی"), WARN,
                     (f"{qpath} not found from the current folder. This path is relative to where you "
                      "run the command, not to the project.",
                      f"مسیر {qpath} از پوشه جاری پیدا نشد؛ این مسیر نسبت به پوشه اجرا است."),
                     ("Run evaluation from the project root, or use an absolute path.",
                      "ارزیابی را از ریشه پروژه اجرا کنید یا مسیر مطلق بدهید."))

    # -- run + render -----------------------------------------------------

    def run_all(self) -> None:
        self.check_python()
        self.check_settings()
        self.check_dependencies()
        self.check_ollama()
        _md, chunk_total, chroma_count, bm25_size = self.check_data()
        self.check_drift(chunk_total, chroma_count, bm25_size)
        self.check_config_sanity()

    @property
    def failures(self) -> List[Check]:
        return [c for c in self.checks if c.status == FAIL]

    @property
    def warnings(self) -> List[Check]:
        return [c for c in self.checks if c.status == WARN]

    def verdict(self) -> Bi:
        if self.failures:
            return (f"NOT READY - {len(self.failures)} blocking problem(s), {len(self.warnings)} warning(s).",
                    f"آماده نیست - {len(self.failures)} مشکل مسدودکننده، {len(self.warnings)} هشدار.")
        if self.warnings:
            return (f"USABLE - no blocking problems, but {len(self.warnings)} thing(s) worth fixing.",
                    f"قابل استفاده - مشکل مسدودکننده‌ای نیست، ولی {len(self.warnings)} مورد بهتر است اصلاح شود.")
        return ("READY - every check passed.", "آماده - همه بررسی‌ها قبول شدند.")

    def next_command(self) -> Bi:
        """One concrete command, chosen from what is actually missing."""
        def failed(fragment: str) -> bool:
            return any(fragment in c.name[0] for c in self.failures)
        if failed("Python version"):
            return ("Install Python 3.10+ first.", "اول پایتون 3.10+ را نصب کنید.")
        if any(c.group[0] == "Dependencies" for c in self.failures):
            return ("pip install -r rag_pipeline/requirements.txt", "pip install -r rag_pipeline/requirements.txt")
        if failed("Ollama server"):
            return ("ollama serve", "ollama serve")
        if any(c.name[0].startswith("models.") for c in self.failures):
            tags = [c.detail[0].split(" - ")[0] for c in self.failures if c.name[0].startswith("models.")]
            return (f"ollama pull {tags[0]}", f"ollama pull {tags[0]}")
        if failed("Input .md files"):
            return ("Copy your OCR .md files into the input folder, then run rag_pipeline/run_pipeline.py",
                    "فایل‌های md را در پوشه ورودی بگذارید و run_pipeline.py را اجرا کنید")
        for frag, cmd in (("Chunks on disk", "python rag_pipeline/run_pipeline.py"),
                          ("Vector database", "python rag_pipeline/run_ingestion.py"),
                          ("Keyword index", "python rag_pipeline/run_bm25_index.py")):
            if any(frag in c.name[0] for c in self.warnings):
                return (cmd, cmd)
        return ('python rag_pipeline/run_answer.py --question "your question here"',
                'python rag_pipeline/run_answer.py --question "سوال شما"')

    def console(self) -> str:
        lines = ["", "=" * 78,
                 "  RAG PIPELINE DOCTOR - preflight check (reads only, changes nothing)",
                 "=" * 78]
        current = None
        for c in self.checks:
            if c.group[0] != current:
                current = c.group[0]
                lines.append(f"\n{current} / {c.group[1]}")
                lines.append("-" * 78)
            lines.append(f"  {c.status} {c.name[0]:<26} {c.detail[0]}")
            if c.fix:
                lines.append(f"      -> fix: {c.fix[0]}")
        v = self.verdict()
        n = self.next_command()
        lines += ["", "=" * 78, f"  VERDICT: {v[0]}", f"  نتیجه:   {v[1]}",
                  f"  NEXT:    {n[0]}", "=" * 78, ""]
        return "\n".join(lines)

    def markdown(self, lang: str = i18n.BOTH) -> str:
        out = ["# Doctor report — گزارش سلامت سیستم\n"]
        out.append(i18n.para((
            "This is a preflight check. It reads your machine, your settings and your data, and "
            "changes nothing. Fix the \u274c rows before trusting any answer the system gives you.",
            "این یک بررسی پیش از اجراست. فقط می‌خواند و چیزی را تغییر نمی‌دهد. پیش از اعتماد به پاسخ‌ها، ردیف‌های \u274c را درست کنید.",
        ), lang))
        v, n = self.verdict(), self.next_command()
        out.append(i18n.table(["Field", "Value"], [
            ["Verdict", v[0]],
            ["نتیجه", v[1]],
            ["Next command", f"`{n[0]}`"],
            ["Checks run", len(self.checks)],
            ["Blocking problems", len(self.failures)],
            ["Warnings", len(self.warnings)],
            ["settings.yaml", str(self.settings_path)],
        ]))
        current = None
        for c in self.checks:
            if c.group[0] != current:
                current = c.group[0]
                out.append(i18n.heading(c.group, 2, lang))
                out.append("| | Check | Result | Fix |")
                out.append("|---|---|---|---|")
            fix = i18n.esc(c.fix[0]) if c.fix else ""
            out.append(f"| {c.status} | {i18n.esc(c.name[0])} | {i18n.esc(c.detail[0])} | {fix} |")
        return "\n".join(out)


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Preflight check for the RAG pipeline.")
    ap.add_argument("--settings", default=str(DEFAULT_SETTINGS), help="Path to settings.yaml")
    ap.add_argument("--out", default=None, help="Folder to also write DOCTOR.md into")
    ap.add_argument("--lang", default=i18n.BOTH, choices=[i18n.EN, i18n.FA, i18n.BOTH])
    args = ap.parse_args(argv)

    doc = Doctor(settings_path=Path(args.settings))
    doc.run_all()
    print(doc.console())

    if args.out:
        folder = Path(args.out)
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / "DOCTOR.md"
        path.write_text(doc.markdown(lang=args.lang), encoding="utf-8")
        print(f"  wrote {path}\n")

    return 1 if doc.failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
