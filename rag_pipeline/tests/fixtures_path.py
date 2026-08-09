"""
tests/fixtures_path.py

Single source of truth for locating the real-sample fixture files
(Binder2.md, Testing.md, and their _metadata.json siblings) that are
shipped inside tests/fixtures/.

Using a relative path (anchored to this file's own location) instead of
an absolute path makes the real-file tests portable across machines —
they work the same whether run from Windows, macOS, Linux, CI, etc.
"""

from pathlib import Path

FIXTURES_DIR = Path(__file__).parent / "fixtures"

BINDER2_MD            = FIXTURES_DIR / "Binder2.md"
BINDER2_METADATA_JSON = FIXTURES_DIR / "Binder2_metadata.json"
TESTING_MD            = FIXTURES_DIR / "Testing.md"
TESTING_METADATA_JSON = FIXTURES_DIR / "Testing_metadata.json"
