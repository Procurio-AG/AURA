"""OULAD download, loading, and SQLite materialization."""

from __future__ import annotations

import argparse
import sqlite3
import zipfile
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "raw"
DB_PATH = ROOT / "oulad.db"

OULAD_URL = "https://ndownloader.figshare.com/files/8606371"
TABLE_NAMES = [
    "courses",
    "assessments",
    "vle",
    "studentInfo",
    "studentRegistration",
    "studentAssessment",
    "studentVle",
]

DTYPES = {
    "courses": {
        "code_module": "string",
        "code_presentation": "string",
        "module_presentation_length": "int64",
    },
    "assessments": {
        "code_module": "string",
        "code_presentation": "string",
        "id_assessment": "int64",
        "assessment_type": "string",
        "date": "float64",
        "weight": "float64",
    },
    "vle": {
        "id_site": "int64",
        "code_module": "string",
        "code_presentation": "string",
        "activity_type": "string",
        "week_from": "float64",
        "week_to": "float64",
    },
    "studentInfo": {
        "code_module": "string",
        "code_presentation": "string",
        "id_student": "int64",
        "gender": "string",
        "region": "string",
        "highest_education": "string",
        "imd_band": "string",
        "age_band": "string",
        "num_of_prev_attempts": "int64",
        "studied_credits": "int64",
        "disability": "string",
        "final_result": "string",
    },
    "studentRegistration": {
        "code_module": "string",
        "code_presentation": "string",
        "id_student": "int64",
        "date_registration": "float64",
        "date_unregistration": "float64",
    },
    "studentAssessment": {
        "id_assessment": "int64",
        "id_student": "int64",
        "date_submitted": "int64",
        "is_banked": "int64",
        "score": "float64",
    },
    "studentVle": {
        "code_module": "string",
        "code_presentation": "string",
        "id_student": "int64",
        "id_site": "int64",
        "date": "int64",
        "sum_click": "int64",
    },
}


def download_oulad(data_dir: Path = DATA_DIR) -> Path:
    data_dir.mkdir(parents=True, exist_ok=True)
    if all((data_dir / f"{name}.csv").exists() for name in TABLE_NAMES):
        print(f"OULAD already present in {data_dir}")
        return data_dir
    zip_path = data_dir / "oulad.zip"
    print(f"Downloading OULAD from {OULAD_URL} ...")
    with requests.get(OULAD_URL, stream=True, timeout=300) as r:
        r.raise_for_status()
        with open(zip_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 20):
                f.write(chunk)
    print(f"Downloaded {zip_path.stat().st_size / 1e6:.1f} MB, extracting ...")
    with zipfile.ZipFile(zip_path) as zf:
        for member in zf.namelist():
            if member.lower().endswith(".csv"):
                target = data_dir / Path(member).name
                with zf.open(member) as src, open(target, "wb") as dst:
                    dst.write(src.read())
    missing = [n for n in TABLE_NAMES if not (data_dir / f"{n}.csv").exists()]
    if missing:
        raise RuntimeError(f"Extraction incomplete, missing: {missing}")
    zip_path.unlink()
    print(f"Extracted {len(TABLE_NAMES)} tables to {data_dir}")
    return data_dir


def load_tables(data_dir: Path = DATA_DIR) -> dict[str, pd.DataFrame]:
    tables: dict[str, pd.DataFrame] = {}
    for name in TABLE_NAMES:
        path = data_dir / f"{name}.csv"
        if not path.exists():
            raise FileNotFoundError(
                f"{path} not found. Run `python -m src.data_loader --download` first."
            )
        df = pd.read_csv(path)
        df = df.astype(DTYPES[name])
        tables[name] = df
        print(f"loaded {name}: {df.shape[0]:,} rows")
    return tables


def build_sqlite(
    tables: dict[str, pd.DataFrame],
    db_path: Path = DB_PATH,
    weekly: bool = True,
) -> Path:
    from src.features import week_index

    if db_path.exists():
        db_path.unlink()
    con = sqlite3.connect(db_path)
    try:
        for name, df in tables.items():
            df.to_sql(name, con, index=False, if_exists="replace")
            print(f"wrote table {name}: {df.shape[0]:,} rows")
        if weekly:
            sv = tables["studentVle"]
            sv = sv.assign(week=week_index(sv["date"]))
            keys = ["code_module", "code_presentation", "id_student"]
            weekly_clicks = sv.groupby(keys + ["week"], as_index=False)["sum_click"].sum()
            weekly_clicks.to_sql("weekly_clicks", con, index=False, if_exists="replace")
            print(f"wrote table weekly_clicks: {weekly_clicks.shape[0]:,} rows")
            vle_map = tables["vle"][["id_site", "activity_type"]].drop_duplicates("id_site")
            sv_a = sv.merge(vle_map, on="id_site", how="left")
            sv_a["activity_type"] = sv_a["activity_type"].fillna("other")
            weekly_activity = sv_a.groupby(
                keys + ["week", "activity_type"], as_index=False
            )["sum_click"].sum()
            weekly_activity.to_sql(
                "weekly_activity", con, index=False, if_exists="replace"
            )
            print(f"wrote table weekly_activity: {weekly_activity.shape[0]:,} rows")
    finally:
        con.close()
    print(f"database ready at {db_path}")
    return db_path


def main() -> None:
    parser = argparse.ArgumentParser(description="OULAD ingestion")
    parser.add_argument("--download", action="store_true", help="download OULAD from figshare")
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--db", type=Path, default=DB_PATH)
    parser.add_argument("--no-weekly", action="store_true", help="skip weekly aggregate tables")
    args = parser.parse_args()
    if args.download:
        download_oulad(args.data_dir)
    tables = load_tables(args.data_dir)
    build_sqlite(tables, args.db, weekly=not args.no_weekly)


if __name__ == "__main__":
    main()
