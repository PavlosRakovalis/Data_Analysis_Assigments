"""
Extract tables from economic_analysis_EAP.pdf using tabula-py.
Runs in both 'lattice' and 'stream' modes and saves every detected table
as an individual CSV plus a run-level report.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import pandas as pd
import tabula

PDF_PATH = Path(__file__).resolve().parent.parent / "economic_analysis_EAP.pdf"
OUT_DIR = Path(__file__).resolve().parent / "tabula_output"


def run_mode(mode: str) -> dict:
    mode_dir = OUT_DIR / mode
    mode_dir.mkdir(parents=True, exist_ok=True)

    kwargs = {"pages": "all", "multiple_tables": True, "pandas_options": {"header": None}}
    if mode == "lattice":
        kwargs["lattice"] = True
    else:
        kwargs["stream"] = True

    start = time.perf_counter()
    tables = tabula.read_pdf(str(PDF_PATH), **kwargs)
    elapsed = time.perf_counter() - start

    # tabula-py does not expose the originating page per table, so we try to
    # re-detect per-page to annotate with it.
    per_page_counts = {}
    try:
        from pypdf import PdfReader

        n_pages = len(PdfReader(str(PDF_PATH)).pages)
    except Exception:
        n_pages = 25

    pages_for_tables: list[int | None] = [None] * len(tables)
    # Re-run per-page to attribute a page number to each extracted table.
    idx = 0
    for p in range(1, n_pages + 1):
        kw = dict(kwargs)
        kw["pages"] = str(p)
        try:
            page_tables = tabula.read_pdf(str(PDF_PATH), **kw)
        except Exception:
            page_tables = []
        per_page_counts[p] = len(page_tables)
        for _ in page_tables:
            if idx < len(pages_for_tables):
                pages_for_tables[idx] = p
                idx += 1

    report = {
        "mode": mode,
        "elapsed_seconds": round(elapsed, 3),
        "num_tables": len(tables),
        "per_page_counts": per_page_counts,
        "tables": [],
    }

    for i, df in enumerate(tables, start=1):
        page = pages_for_tables[i - 1]
        csv_path = mode_dir / f"table_{i:02d}_page{page}.csv"
        df.to_csv(csv_path, index=False, header=False)
        report["tables"].append(
            {
                "index": i,
                "page": page,
                "csv": csv_path.name,
                "shape": list(df.shape),
            }
        )

    (mode_dir / "report.json").write_text(json.dumps(report, indent=2))
    return report


def main() -> None:
    overall = {}
    for mode in ("lattice", "stream"):
        print(f"\n=== Tabula-py ({mode}) ===")
        try:
            r = run_mode(mode)
            overall[mode] = r
            print(f"  tables: {r['num_tables']}  elapsed: {r['elapsed_seconds']}s")
            for t in r["tables"]:
                print(f"    #{t['index']:>2} page={t['page']} shape={t['shape']}")
        except Exception as exc:
            overall[mode] = {"mode": mode, "error": str(exc)}
            print(f"  ERROR: {exc}")

    (OUT_DIR / "summary.json").write_text(json.dumps(overall, indent=2))
    print(f"\nSummary written to {OUT_DIR / 'summary.json'}")


if __name__ == "__main__":
    main()
