import csv
from pathlib import Path

PROCESSED_DIR = Path("data/processed")
META_FILE = PROCESSED_DIR / "metadata.csv"
QRELS_FILE = PROCESSED_DIR / "qrels.csv"

CATEGORY_QUERIES = {
    "business": "business economy market finance companies stocks",
    "politics": "politics government election parliament policy",
    "sport": "sports football cricket match league competition",
    "entertainment": "entertainment movie film music celebrity show",
    "tech": "technology innovation science ai computer software"
}

def generate_qrels():
    # Load metadata
    rows = []
    with META_FILE.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)

    # Group filenames by category
    category_docs = {}
    for r in rows:
        cat = r["category"]
        fname = r["filename"]
        category_docs.setdefault(cat, []).append(fname)

    # Prepare qrels rows
    qrels_rows = []
    query_id = 1

    for category, files in category_docs.items():
        query_text = CATEGORY_QUERIES.get(category, category)

        qrels_rows.append({
            "query_id": query_id,
            "query_text": query_text,
            "relevant_filenames": "|".join(files)
        })
        query_id += 1

    # Save qrels.csv
    with QRELS_FILE.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["query_id", "query_text", "relevant_filenames"]
        )
        writer.writeheader()
        for row in qrels_rows:
            writer.writerow(row)

    print(f"qrels.csv generated successfully → {QRELS_FILE}")


if __name__ == "__main__":
    generate_qrels()
