#!/usr/bin/env python3
"""CLI utility to ingest OULAD CSVs and build oulad.db SQLite database.

Usage:
    python scripts/ingest_data.py --download   # Downloads from figshare and builds SQLite DB
    python scripts/ingest_data.py              # Ingests from data/raw and builds SQLite DB
"""

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_loader import ingest, DEFAULT_DATA_DIR, DEFAULT_DB_PATH

def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="OULAD Ingestion & SQLite Builder")
    parser.add_argument("--download", action="store_true", help="Download raw CSVs from Figshare")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR, help="Directory containing raw OULAD CSVs")
    parser.add_argument("--db", type=Path, default=DEFAULT_DB_PATH, help="Path for output oulad.db SQLite database")
    parser.add_argument("--no-weekly", action="store_true", help="Skip weekly aggregate tables")
    args = parser.parse_args()

    print(f"[*] Starting ingestion pipeline...")
    print(f"[*] Raw data directory: {args.data_dir}")
    print(f"[*] Target SQLite DB:  {args.db}")

    if not args.download and not any(args.data_dir.glob("*.csv")):
        print(f"[!] Warning: No CSV files found in {args.data_dir}.")
        print("[!] Pass '--download' to automatically download OULAD from figshare, or manually place CSV files in data/raw/.")
        sys.exit(1)

    db_path = ingest(download=args.download, data_dir=args.data_dir, db_path=args.db, weekly=not args.no_weekly)
    print(f"[✓] Pipeline complete! Database created at: {db_path}")

if __name__ == "__main__":
    main()
