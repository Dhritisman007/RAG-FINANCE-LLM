"""
Day 18: Tier 2 India — annual reports of large NSE-listed companies.

For each company in config/corpus_manifest.yaml (tier2_india):
  1. Ask NSE's public annual-report listing for the latest report
  2. Download the PDF (cached in data/pdf_cache/india/, git-ignored)
  3. Extract text with pypdf and label each page with its section, read from
     the page's running header ("Board's Report", "Consolidated Financial
     Statements", ...). Keep the integrated report, Board's report, MD&A,
     corporate governance and the CONSOLIDATED statements; skip the AGM notice,
     the BRSR and the standalone statements (parent-company-only figures that
     a model could quote as the company's results)
  4. Repair the rupee symbol (report fonts map it to H, I, C, K or `), then
     save .txt + .meta.json in data/raw/tier2/ — the same schema as the SEC
     filings, so the chunker, enrichment and vector store handle them unchanged

Usage:
    python scripts/scrape_tier2_india.py                   # all companies (skips reports already on disk)
    python scripts/scrape_tier2_india.py --symbol RELIANCE # one company
    python scripts/scrape_tier2_india.py --reextract       # re-extract cached PDFs (no downloads)
"""
from __future__ import annotations

import argparse
import io
import json
import re
import time
import zipfile
from collections import Counter
from datetime import date
from pathlib import Path

import requests
import yaml

ROOT = Path(__file__).parent.parent
MANIFEST_PATH = ROOT / "config" / "corpus_manifest.yaml"
OUT_DIR = ROOT / "data" / "raw" / "tier2"
PDF_DIR = ROOT / "data" / "pdf_cache" / "india"

LISTING_URL = "https://www.nseindia.com/api/annual-reports?index=equities&symbol={symbol}"
# An honest label, no browser impersonation: if NSE starts refusing this, stop and use manual PDFs.
HEADERS = {"User-Agent": "FinRAG/0.1 (personal research; local RAG project)"}
REQUEST_DELAY_SEC = 1.0

DEFAULT_SECTION = "Integrated Report"
# (section name, header pattern, keep?) — checked against each page's first lines, in order.
SECTIONS = [
    # BRSR pages are headed by the report name or its "Essential/Leadership indicators" (SEBI format);
    # SBI prints a separate "Sustainability Report" inside its annual report.
    ("Business Responsibility and Sustainability Report",
     r"business responsibility|\bbrsr\b|^sustainability report\b|essential indicators|leadership indicators", False),
    ("Standalone Financial Statements", r"standalone financial(?:s\b|\s+statements?)", False),   # SBI: "STANDALONE FINANCIALS"
    ("Consolidated Financial Statements", r"consolidated financial(?:s\b|\s+statements?)", True),
    # AGM notice — also its explanatory statement ("Item No. 6 – Material Related Party Transactions"),
    # which some reports print after the financial statements
    ("Notice", r"^notice\b(?! of)|^notice of (the )?annual general meeting|^item no\.?\s*\d+\s*[–-]", False),
    ("Management Discussion and Analysis", r"management['’]?s? discussion (and|&) analysis", True),
    ("Board's Report", r"^(the )?(board|directors)['’]?s?['’]? report"
                       r"|board of directors (hereby )?(submits|present)", True),  # Infosys: no header, an opening line
    ("Corporate Governance Report", r"corporate governance report|report on corporate governance", True),
]
HEADER_LINES = 5  # how many lines at the top of a page count as its running header
# Some reports (Asian Paints) mark statement pages only with a footer:
# "to the Consolidated Financial Statements for the year ended ... (Contd.)". Only this exact
# form is read from the bottom of a page, so prose that ends a page can't relabel it.
FOOTER_RE = re.compile(r"^(?:notes\s+)?to the (standalone|consolidated) financial statements", re.I)

# The rupee glyph extracts as a stray capital/backtick right before an amount in crore/lakh.
RUPEE_AMOUNT_RE = re.compile(r"(?<![A-Za-z0-9])[HICK`](?=\s?\d[\d,]*(?:\.\d+)?\s*(?:crore|lakh|cr\b))")
# "(H in crore)" and "(In ` crore, except per share data)" — the glyph before or after "in"
RUPEE_UNIT_RE = re.compile(r"\((?:\s*[HICK`₹]\s*(?i:in)|\s*(?i:in)\s+[HICK`₹])\s+((?i:crore|lakh|million|billion))s?")
PAGE_NUMBER_RE = re.compile(r"^\d{1,3}(?:\s+\d{1,3})?$")
# Major subsidiaries' summary statements sit inside the consolidated section, headed by the
# subsidiary's name ("Jio Platforms Limited"). A 7B model quoted Jio's revenue as Reliance's,
# so these pages are labelled as the subsidiary's.
COMPANY_LINE_RE = re.compile(r"^(?:consolidated financial statements\s*)?"
                             r"([A-Z][A-Za-z0-9&.,'() -]{2,70}?\b(?:Limited|Ltd\.?|LLC|Inc\.?|GmbH|B\.V\.|Pte\.? Ltd\.?))"
                             r"\s*\d{0,3}$", re.I)


def subsidiary_name(lines: list[str], parent: str) -> str | None:
    """A company named in the page header that isn't the parent (whitespace normalised)."""
    for line in lines[:2]:  # the statement's own header; a name further down is just mentioned
        m = COMPANY_LINE_RE.match(re.sub(r"\s+", " ", line))
        if m and parent.lower() not in m.group(1).lower():
            return m.group(1)
    return None


def load_companies() -> list[dict]:
    with open(MANIFEST_PATH) as f:
        return yaml.safe_load(f)["tier2_india"]["companies"]


def latest_report(symbol: str, get=requests.get) -> dict | None:
    """NSE's newest annual-report entry for `symbol` (highest fiscal year)."""
    resp = get(LISTING_URL.format(symbol=symbol), headers=HEADERS, timeout=30)
    resp.raise_for_status()
    rows = [r for r in (resp.json() or {}).get("data", []) if r.get("fileName")]
    return max(rows, key=lambda r: int(r["toYr"])) if rows else None


def download_pdf(url: str, dest: Path) -> Path:
    """Download (or reuse) the report PDF; a .zip is unpacked to its largest PDF."""
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    resp = requests.get(url, headers=HEADERS, timeout=300)
    resp.raise_for_status()
    data = resp.content
    if url.lower().endswith(".zip"):
        with zipfile.ZipFile(io.BytesIO(data)) as z:
            pdfs = [i for i in z.infolist() if i.filename.lower().endswith(".pdf")]
            if not pdfs:
                raise RuntimeError(f"no PDF inside {url}")
            data = z.read(max(pdfs, key=lambda i: i.file_size))
    if not data.startswith(b"%PDF"):
        raise RuntimeError(f"not a PDF: {url}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(data)
    return dest


# ── Side-by-side Standalone | Consolidated tables (Day 19) ─────────────────
# PDF text flattens them into rows of bare numbers ("Revenue from operations 1,48,819
# 1,36,592 9.0 1,78,650 1,62,990 9.6") and a 7B model quoted the first, standalone,
# column as Infosys' revenue. Every such row is rewritten with its values labelled.
# The standalone block's columns always precede the consolidated block's (or the
# reverse, per the header), so a row with 2m values splits into halves; a period line
# ("FY2026 FY2025 Growth FY2026 FY2025 Growth") names each value. Rows with an odd
# count are left as they are rather than guessed.
DUAL_HEADER_RE = re.compile(r"\b(standalone|consolidated)\b(?:(?!\band\b).)*\b(consolidated|standalone)\s*$", re.I)
CELL = r"\(?-?[\d,]*\d(?:\.\d+)?%?\)?|[-–—]"
ROW_RE = re.compile(rf"^(?P<label>.*?[A-Za-z)\].:].*?)\s+(?P<cells>(?:(?:{CELL})\s+)+(?:{CELL}))\s*$")
PERIOD_TOKEN_RE = re.compile(r"FY\s?\d{4}(?:-\d{2})?|FY\s?\d{2}\b|\b(?:19|20)\d{2}-\d{2}\b|\b(?:19|20)\d{2}\b|\(%\)|"
                             r"\bGrowth\b|\bYoY\b|\bchange\b", re.I)
# Inside a table row "29, 211" is one number with a stray space from the PDF (Infosys'
# standalone profit); in prose ("in 2026, 220 clients") it isn't, so this applies to rows only.
SPLIT_NUMBER_RE = re.compile(r"(\d), (\d{2,3}(?:,\d{2,3})*(?:\.\d+)?)(?=\s|$)")
TABLE_SPAN = 40     # lines after a header that can belong to its table
TABLE_GAP = 6       # consecutive non-row lines that end a table once rows have started


def _period_labels(line: str) -> list[str] | None:
    """Column labels if the line is (almost) only period/growth tokens, e.g. '2026 2025 (%) 2026 2025 (%)'."""
    tokens = PERIOD_TOKEN_RE.findall(line)
    rest = PERIOD_TOKEN_RE.sub("", line).strip(" ,|")
    if len(tokens) >= 2 and len(rest) <= 3:
        return ["YoY %" if tok.strip() == "(%)" else re.sub(r"\s+", " ", tok.strip()) for tok in tokens]
    return None


def label_dual_tables(lines: list[str]) -> list[str]:
    """Rewrite rows of Standalone|Consolidated tables with labelled values (see above)."""
    out = list(lines)
    i = 0
    while i < len(out):
        m = DUAL_HEADER_RE.search(out[i])
        if not m:
            i += 1
            continue
        first, second = m.group(1).title(), m.group(2).title()
        if first == second:
            i += 1
            continue
        labels, gap, started = None, 0, False
        j = i + 1
        while j < len(out) and j <= i + TABLE_SPAN and gap < TABLE_GAP:
            if DUAL_HEADER_RE.search(out[j]) or out[j].startswith("## "):
                break
            if labels is None and (lab := _period_labels(out[j])):
                labels = lab
                j += 1
                continue
            row = ROW_RE.match(SPLIT_NUMBER_RE.sub(r"\1,\2", out[j]))
            cells = row.group("cells").split() if row else []
            if row and len(cells) >= 2 and len(cells) % 2 == 0:
                half = len(cells) // 2
                cols = labels if labels and len(labels) == len(cells) else None
                if labels and cols is None:  # a row without its growth cells: the period labels still apply
                    periods = [lab for lab in labels if lab.lower() not in ("yoy %", "growth", "yoy", "change")]
                    cols = periods if len(periods) == len(cells) else None

                def block(vals, offset):
                    return ", ".join(f"{cols[offset + k]} {v}" for k, v in enumerate(vals)) if cols else " ".join(vals)
                out[j] = (f"{row.group('label').rstrip(' :')} — {first}: {block(cells[:half], 0)}; "
                          f"{second}: {block(cells[half:], half)}")
                started, gap = True, 0
            elif started:
                gap += 1
            j += 1
        i = j
    return out


def clean_text(text: str) -> str:
    text = RUPEE_UNIT_RE.sub(lambda m: f"(₹ in {m.group(1).lower()}", text)
    text = RUPEE_AMOUNT_RE.sub("₹", text)
    text = text.replace("\t", " ")
    text = re.sub(r"(?m)^\s*#+", "*", text)  # footnote markers ("## ...") would read as section headings
    text = re.sub(r"[  ]{2,}", " ", text)
    return text


def page_section(lines: list[str]) -> str | None:
    """The section named in a page's running header (or financial-statement footer), if any."""
    for line in lines[:HEADER_LINES]:
        for name, pattern, _ in SECTIONS:
            if re.search(pattern, line, re.I):
                return name
    for line in lines[-HEADER_LINES:]:
        m = FOOTER_RE.match(line)
        if m:
            return f"{m.group(1).title()} Financial Statements"
    return None


def build_document(pages: list[str], parent: str = "") -> tuple[str, Counter]:
    """
    Pages -> text with "## <section> (p. N)" headings for kept sections only.
    Returns the text and the number of pages kept/skipped per section.
    """
    page_lines = [[l.strip() for l in p.split("\n") if l.strip()] for p in pages]
    # Running headers repeat on many pages; drop them from the body text.
    counts = Counter(l for lines in page_lines for l in set(lines[:HEADER_LINES]))
    repeated = {l for l, n in counts.items() if n >= max(5, len(pages) // 10) and len(l) <= 120}
    keep = {name: k for name, _, k in SECTIONS} | {DEFAULT_SECTION: True}

    section, last_kept, anchor, out, stats = DEFAULT_SECTION, DEFAULT_SECTION, None, [], Counter()
    for i, lines in enumerate(page_lines, start=1):
        explicit = page_section(lines)
        if explicit:
            section = explicit
            # The running header this section's pages share (e.g. the notice's "Reliance Industries Limited").
            anchor = next((l for l in lines[:HEADER_LINES]
                           if not page_section([l]) and not PAGE_NUMBER_RE.match(l)), None)
        elif not keep[section] and anchor not in lines[:HEADER_LINES]:
            # A skipped section (notice, BRSR, standalone) only continues while its header does —
            # otherwise a notice printed before the integrated report would swallow it.
            section = last_kept
        if keep[section]:
            last_kept = section
        stats[(section, keep[section])] += 1
        if not keep[section]:
            continue
        printed = next((l for l in lines[:HEADER_LINES] if PAGE_NUMBER_RE.match(l)), None)
        label = f"p. {printed.replace(' ', '–')}" if printed else f"PDF page {i}"
        body = [l for j, l in enumerate(lines)
                if not (j < HEADER_LINES and (l in repeated or PAGE_NUMBER_RE.match(l)))]
        if sum(len(l) for l in body) < 200:  # cover art, photo pages, dividers
            continue
        subsidiary = subsidiary_name(lines, parent) if parent and section == "Consolidated Financial Statements" else None
        heading = f"{section} — Subsidiary: {subsidiary}" if subsidiary else section
        out.append(f"## {heading} ({label})")
        out.append(clean_text("\n".join(label_dual_tables(body))))
    return "\n\n".join(out), stats


def extract_pages(pdf_path: Path) -> list[str]:
    from pypdf import PdfReader
    return [page.extract_text() or "" for page in PdfReader(str(pdf_path)).pages]


def slug(symbol: str, report: dict) -> str:
    return f"tier2_nse_{symbol}-AR-FY{report['fromYr']}-{report['toYr'][-2:]}"


def save_doc(company: dict, report: dict, text: str) -> Path:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    s = slug(company["symbol"], report)
    fy = f"FY{report['fromYr']}-{report['toYr'][-2:]}"
    (OUT_DIR / f"{s}.txt").write_text(text, encoding="utf-8")
    published = report.get("broadcast_dttm") or ""
    try:
        from datetime import datetime
        published = datetime.strptime(published, "%d-%b-%Y %H:%M:%S").date().isoformat()
    except ValueError:
        published = None
    metadata = {
        "source": "nse_annual_report",
        "tier": 2,
        "doc_type": "annual_report",
        "title": f"{company['name']} ({company['symbol']}) Annual Report {fy}",
        "url": report["fileName"],
        "date_fetched": date.today().isoformat(),
        "date_published": published,
        "company_ticker": f"{company['symbol']}.NS",
        "arxiv_id": None,
        "char_count": len(text),
    }
    (OUT_DIR / f"{s}.meta.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    return OUT_DIR / f"{s}.txt"


def process(company: dict, reextract: bool = False) -> None:
    symbol = company["symbol"]
    print(f"\n{symbol} — {company['name']}")
    report = latest_report(symbol)
    time.sleep(REQUEST_DELAY_SEC)
    if report is None:
        print("  [SKIP] NSE lists no annual report")
        return
    txt = OUT_DIR / f"{slug(symbol, report)}.txt"
    if txt.exists() and not reextract:
        print(f"  [EXISTS] {txt.name}")
        return
    pdf = PDF_DIR / f"{symbol}_{report['fromYr']}_{report['toYr']}.pdf"
    if not pdf.exists():
        print(f"  downloading FY{report['fromYr']}-{report['toYr'][-2:]} ({report.get('attFileSize') or '?'})…")
    download_pdf(report["fileName"], pdf)
    text, stats = build_document(extract_pages(pdf), parent=company["name"])
    kept = sum(n for (_, k), n in stats.items() if k)
    skipped = ", ".join(f"{name} {n}p" for (name, k), n in sorted(stats.items()) if not k)
    path = save_doc(company, report, text)
    print(f"  [SAVED] {path.name} — {len(text):,} chars from {kept} pages (skipped: {skipped or 'nothing'})")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download and extract Indian annual reports (NSE).")
    parser.add_argument("--symbol", help="Only this NSE symbol, e.g. RELIANCE")
    parser.add_argument("--reextract", action="store_true", help="Re-extract even if the .txt exists")
    args = parser.parse_args()
    companies = [c for c in load_companies() if not args.symbol or c["symbol"] == args.symbol.upper()]
    failed = []
    for c in companies:
        try:
            process(c, args.reextract)
        except Exception as e:  # one bad report shouldn't stop the rest
            print(f"  [FAILED] {e}")
            failed.append(c["symbol"])
    print(f"\nDone: {len(companies) - len(failed)}/{len(companies)}" + (f" — failed: {', '.join(failed)}" if failed else ""))


if __name__ == "__main__":
    main()
