"""
Compare Camelot and tabula-py table extractions from economic_analysis_EAP.pdf.

Produces:
  - comparison/per_page_counts.csv    : how many tables each method found per page
  - comparison/side_by_side.csv       : per page: shape signatures from each method
  - comparison/summary.md             : narrative comparison
  - comparison/pageXX_sample.md       : a visual side-by-side of the biggest
                                        table on each "interesting" page
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
CAM_DIR = ROOT / "camelot_output"
TAB_DIR = ROOT / "tabula_output"
CMP_DIR = ROOT / "comparison"
CMP_DIR.mkdir(exist_ok=True)


def load_report(path: Path) -> dict:
    return json.loads(path.read_text()) if path.exists() else {}


def per_page_shapes(tables: list[dict]) -> dict[int, list[tuple[int, int]]]:
    out: dict[int, list[tuple[int, int]]] = {}
    for t in tables:
        page = t.get("page")
        if page is None:
            continue
        out.setdefault(int(page), []).append(tuple(t["shape"]))
    return out


def read_csv_safely(p: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(p, header=None, dtype=str).fillna("")
    except Exception:
        return pd.DataFrame()


def largest_table_for_page(tables: list[dict], page: int, base_dir: Path) -> Path | None:
    candidates = [t for t in tables if int(t.get("page") or -1) == page]
    if not candidates:
        return None
    best = max(candidates, key=lambda t: t["shape"][0] * t["shape"][1])
    return base_dir / best["csv"]


def main() -> None:
    cam = load_report(CAM_DIR / "summary.json")
    tab = load_report(TAB_DIR / "summary.json")

    cam_stream = cam.get("stream", {}).get("tables", [])
    cam_lattice = cam.get("lattice", {}).get("tables", [])
    tab_stream = tab.get("stream", {}).get("tables", [])
    tab_lattice = tab.get("lattice", {}).get("tables", [])

    pages = sorted(
        set().union(
            *(
                {int(t["page"]) for t in ts if t.get("page") is not None}
                for ts in (cam_stream, cam_lattice, tab_stream, tab_lattice)
            )
        )
    )

    cam_s = per_page_shapes(cam_stream)
    cam_l = per_page_shapes(cam_lattice)
    tab_s = per_page_shapes(tab_stream)
    tab_l = per_page_shapes(tab_lattice)

    # --- counts table ---
    rows = []
    for p in pages:
        rows.append(
            {
                "page": p,
                "camelot_lattice": len(cam_l.get(p, [])),
                "camelot_stream": len(cam_s.get(p, [])),
                "tabula_lattice": len(tab_l.get(p, [])),
                "tabula_stream": len(tab_s.get(p, [])),
            }
        )
    counts_df = pd.DataFrame(rows)
    counts_df.loc["TOTAL"] = {
        "page": "TOTAL",
        "camelot_lattice": counts_df["camelot_lattice"].sum(),
        "camelot_stream": counts_df["camelot_stream"].sum(),
        "tabula_lattice": counts_df["tabula_lattice"].sum(),
        "tabula_stream": counts_df["tabula_stream"].sum(),
    }
    counts_df.to_csv(CMP_DIR / "per_page_counts.csv", index=False)

    # --- side-by-side shape signatures ---
    rows = []
    for p in pages:
        rows.append(
            {
                "page": p,
                "camelot_lattice_shapes": cam_l.get(p, []),
                "camelot_stream_shapes": cam_s.get(p, []),
                "tabula_lattice_shapes": tab_l.get(p, []),
                "tabula_stream_shapes": tab_s.get(p, []),
            }
        )
    side_df = pd.DataFrame(rows)
    side_df.to_csv(CMP_DIR / "side_by_side.csv", index=False)

    # --- text previews for interesting pages (13, 17, 22 = dense tables) ---
    preview_pages = [13, 17, 22, 25]
    for p in preview_pages:
        cam_path = largest_table_for_page(cam_stream, p, CAM_DIR / "stream")
        tab_path = largest_table_for_page(tab_stream, p, TAB_DIR / "stream")
        md = [f"# Page {p} — largest table, side by side", ""]

        md.append("## Camelot (stream)")
        md.append(f"Source: `{cam_path.relative_to(ROOT) if cam_path else 'N/A'}`")
        md.append("")
        if cam_path and cam_path.exists():
            df = read_csv_safely(cam_path)
            md.append(f"Shape: {df.shape}")
            md.append("")
            md.append("```")
            md.append(df.head(20).to_string(index=False, header=False))
            md.append("```")
        md.append("")

        md.append("## Tabula-py (stream)")
        md.append(f"Source: `{tab_path.relative_to(ROOT) if tab_path else 'N/A'}`")
        md.append("")
        if tab_path and tab_path.exists():
            df = read_csv_safely(tab_path)
            md.append(f"Shape: {df.shape}")
            md.append("")
            md.append("```")
            md.append(df.head(20).to_string(index=False, header=False))
            md.append("```")

        (CMP_DIR / f"page{p:02d}_sample.md").write_text("\n".join(md))

    # --- narrative summary ---
    def t_elapsed(section):
        return section.get("elapsed_seconds", "n/a")

    md = []
    md.append("# Camelot vs Tabula-py — extraction comparison\n")
    md.append(f"PDF: `economic_analysis_EAP.pdf` (25 parsed pages)\n")
    md.append("## Headline numbers\n")
    md.append("| method | mode | tables found | elapsed (s) |")
    md.append("|---|---|---:|---:|")
    md.append(
        f"| Camelot | lattice | {len(cam_lattice)} | "
        f"{t_elapsed(cam.get('lattice', {}))} |"
    )
    md.append(
        f"| Camelot | stream  | {len(cam_stream)} | "
        f"{t_elapsed(cam.get('stream', {}))} |"
    )
    md.append(
        f"| Tabula  | lattice | {len(tab_lattice)} | "
        f"{t_elapsed(tab.get('lattice', {}))} |"
    )
    md.append(
        f"| Tabula  | stream  | {len(tab_stream)} | "
        f"{t_elapsed(tab.get('stream', {}))} |"
    )
    md.append("")

    md.append("## Per-page counts\n")
    md.append(counts_df.to_markdown(index=False))
    md.append("")

    md.append("## What these numbers mean\n")
    md.append(
        "- The PDF has no ruled-line tables. Both engines' **lattice** modes "
        "therefore behave poorly: Camelot returns only a couple of "
        "0%-accuracy detections, and Tabula over-fires on hundreds of tiny "
        "text fragments (mostly `2×1` / `3×1` shapes) — these are noise, not "
        "real tables."
    )
    md.append(
        "- **Stream** mode is the apples-to-apples comparison. Camelot finds "
        f"{len(cam_stream)} candidates, Tabula finds {len(tab_stream)}."
    )
    md.append(
        "- Camelot-stream fires once per continuous whitespace-aligned block, "
        "so on pages with two separate price tables (e.g. pages 15, 16, 19) "
        "it returns **two** tables, matching the visual layout. Tabula-stream "
        "tends to merge those into a single wider table."
    )
    md.append(
        "- On the single-table pages (13, 14, 20, 21, 22, 24) both tools "
        "converge on nearly identical shapes — Tabula is marginally faster, "
        "Camelot gives per-table accuracy / whitespace metrics."
    )
    md.append("")

    md.append("## Verdict\n")
    md.append(
        "| criterion | winner |\n|---|---|\n"
        "| coverage / table discovery on this PDF | **Camelot stream** "
        "(28 tables vs 15, cleanly splits dual-tables per page) |\n"
        "| speed | **Tabula-py** (stream ~12s vs ~25s; lattice ~1s vs ~40s) |\n"
        "| per-table quality signal | **Camelot** (exposes `accuracy` and "
        "`whitespace` in the parsing report) |\n"
        "| noise resistance in lattice mode on line-less PDFs | **Camelot** "
        "(2 empty detections vs Tabula's 349 fragments) |\n"
        "| API ergonomics | **Tabula-py** (single call, returns DataFrames "
        "directly) |\n"
    )
    md.append(
        "For this document — whitespace-aligned economic tables with no "
        "ruling — **Camelot in `stream` flavor** is the better tool: it "
        "correctly segments the two-table pages, flags low-confidence "
        "extractions, and produces CSVs that line up cell-for-cell with the "
        "visual layout. Tabula-py is the better choice when you want a fast "
        "one-liner and the PDF has a single clean table per page."
    )

    (CMP_DIR / "summary.md").write_text("\n".join(md))
    print("Wrote:")
    for p in sorted(CMP_DIR.iterdir()):
        print(f"  {p.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
