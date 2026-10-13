"""Root data loader adapter defaulting to root project paths."""
from __future__ import annotations
from pathlib import Path
import sys

_FINAL_PATH = Path(__file__).resolve().parent.parent / "final"
if str(_FINAL_PATH) not in sys.path:
    sys.path.insert(0, str(_FINAL_PATH))

from final.src.data_loader import (  # noqa: F401
    TABLE_NAMES,
    DTYPES,
    OULAD_URL,
    download_oulad,
    load_tables,
    build_sqlite,
)

ROOT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_DATA_DIR = ROOT_DIR / "data" / "raw"
DEFAULT_DB_PATH = ROOT_DIR / "oulad.db"

def ingest(download: bool = False, data_dir: Path | None = None, db_path: Path | None = None, weekly: bool = True) -> Path:
    """Run full OULAD ingestion and SQLite database materialization."""
    target_data = Path(data_dir) if data_dir else DEFAULT_DATA_DIR
    target_db = Path(db_path) if db_path else DEFAULT_DB_PATH
    if download:
        download_oulad(target_data)
    tables = load_tables(target_data)
    return build_sqlite(tables, target_db, weekly=weekly)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="OULAD Ingestion Pipeline")
    parser.add_argument("--download", action="store_true", help="Download raw CSVs from figshare")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR, help="Path to raw CSVs")
    parser.add_argument("--db", type=Path, default=DEFAULT_DB_PATH, help="Destination SQLite path")
    parser.add_argument("--no-weekly", action="store_true", help="Skip precomputed weekly aggregations")
    args = parser.parse_args()
    ingest(download=args.download, data_dir=args.data_dir, db_path=args.db, weekly=not args.no_weekly)
