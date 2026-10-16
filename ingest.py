"""
Standalone document ingestion script.
Usage: python ingest.py
"""

import sys
from run import run_ingest

class Args:
    pass

if __name__ == '__main__':
    run_ingest(Args())
