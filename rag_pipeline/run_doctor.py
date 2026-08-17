#!/usr/bin/env python3
"""
run_doctor.py -- START HERE.

Before you run the pipeline, run this. It checks, in plain language:

    Is Python new enough?
    Are the Python packages installed?
    Is Ollama running, and are the exact models in settings.yaml downloaded?
    Do you actually have input documents, chunks, a vector DB and a BM25 index?
    Do those two indexes agree with each other and with the files on disk?
    Are there settings that contradict each other?

It prints a verdict and the single next command to run. It never writes to
your data, never downloads anything, and never loads a model.

    python rag_pipeline/run_doctor.py
    python rag_pipeline/run_doctor.py --out reports     # also write reports/DOCTOR.md
    python rag_pipeline/run_doctor.py --lang fa         # Persian only
"""
import sys
from pathlib import Path

# Allow running from the project root without installing the package.
sys.path.insert(0, str(Path(__file__).parent.parent))

from rag_pipeline.doctor import main

if __name__ == "__main__":
    sys.exit(main())
