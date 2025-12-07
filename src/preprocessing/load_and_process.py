# src/preprocessing/load_and_process.py
from pathlib import Path
from .clean_text import clean_text
import csv

RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

def process_dataset():
    """
    Walks data/raw/category/*.txt and writes cleaned docs to data/processed/{doc_id}.txt
    Also writes data/processed/metadata.csv with columns: doc_id, filename, category, source_path
    """
    meta = []
    doc_id = 0
    for category_dir in sorted(RAW_DIR.iterdir()):
        if not category_dir.is_dir():
            continue
        category = category_dir.name
        for raw_file in sorted(category_dir.glob("*.txt")):
            raw_text = raw_file.read_text(encoding="utf-8", errors="ignore")
            cleaned = clean_text(raw_text)
            out_path = PROCESSED_DIR / f"{doc_id}.txt"
            out_path.write_text(cleaned, encoding="utf-8")
            meta.append({
                "doc_id": doc_id,
                "filename": f"{doc_id}.txt",
                "category": category,
                "source_path": str(raw_file)
            })
            doc_id += 1

    # write metadata CSV
    meta_path = PROCESSED_DIR / "metadata.csv"
    with meta_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["doc_id", "filename", "category", "source_path"])
        writer.writeheader()
        for row in meta:
            writer.writerow(row)

    print(f"Processed {doc_id} documents. Metadata at {meta_path}")

if __name__ == "__main__":
    process_dataset()
