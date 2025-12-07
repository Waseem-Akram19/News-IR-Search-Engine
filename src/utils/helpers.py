# src/utils/helpers.py
from pathlib import Path

def read_processed_doc(filename:str):
    p = Path("data/processed") / filename
    return p.read_text(encoding="utf-8") if p.exists() else ""
