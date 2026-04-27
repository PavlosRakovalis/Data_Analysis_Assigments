"""
Extract tables from economic_analysis_EAP.pdf using Camelot.
Runs in both 'lattice' (ruled-line) and 'stream' (whitespace) modes and
saves every detected table as an individual CSV plus a run-level report.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import camelot
import pandas as pd

PDF_PATH = Path(__file__).resolve().parent.parent / "economic_analysis_EAP.pdf"
OUT_DIR = Path(__file__).resolve().parent / "camelot_output"
PAGES = "1-end"


def run_flavor(flavor: str) -> dict:
    flavor_dir = OUT_DIR / flavor
    flavor_dir.mkdir(parents=True, exist_ok=True)

    start = time.perf_counter()
    tables = camelot.read_pdf(str(PDF_PATH), pages=PAGES, flavor=flavor)
    elapsed = time.perf_counter() - start

    report = {
        "flavor": flavor,
        "elapsed_seconds": round(elapsed, 3),
        "num_tables": len(tables),
        "tables": [],
    }

    for i, t in enumerate(tables, start=1):
        csv_path = flavor_dir / f"table_{i:02d}_page{t.page}.csv"
        t.df.to_csv(csv_path, index=False, header=False)
        info = {
            "index": i,
            "page": t.page,
            "csv": csv_path.name,
            "shape": list(t.df.shape),
            "accuracy": round(float(t.parsing_report.get("accuracy", 0.0)), 2),
            "whitespace": round(float(t.parsing_report.get("whitespace", 0.0)), 2),
            "order": t.parsing_report.get("order"),
        }
        report["tables"].append(info)

    (flavor_dir / "report.json").write_text(json.dumps(report, indent=2))
    return report


def main() -> None:
    overall = {}
    for flavor in ("lattice", "stream"):
        print(f"\n=== Camelot ({flavor}) ===")
        try:
            r = run_flavor(flavor)
            overall[flavor] = r
            print(f"  tables: {r['num_tables']}  elapsed: {r['elapsed_seconds']}s")
            for t in r["tables"]:
                print(
                    f"    #{t['index']:>2} page={t['page']:>3} "
                    f"shape={t['shape']} acc={t['accuracy']} ws={t['whitespace']}"
                )
        except Exception as exc:  # camelot lattice fails hard on pages w/o ruled lines
            overall[flavor] = {"flavor": flavor, "error": str(exc)}
            print(f"  ERROR: {exc}")

    (OUT_DIR / "summary.json").write_text(json.dumps(overall, indent=2))
    print(f"\nSummary written to {OUT_DIR / 'summary.json'}")


if __name__ == "__main__":
    main()
