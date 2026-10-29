"""
Indian annual-report extraction tests. No network, no PDFs: pages are synthetic.

    python tests/test_india_reports.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

import scrape_tier2_india as india  # noqa: E402

HEADER = "Acme Limited Integrated Annual Report 2025-26"
FILLER = "The company reported steady growth across its segments during the year. " * 4


def page(*lines: str) -> str:
    return "\n".join(lines)


def test_rupee_symbol_repair() -> None:
    raw = "Revenue was I 9,10,000 crore and capex C 1,32,500 crore; dividend H5.50 lakh.\n(H in crore)\n(` in Crores)"
    out = india.clean_text(raw)
    assert "₹ 9,10,000 crore" in out and "₹ 1,32,500 crore" in out and "₹5.50 lakh" in out, out
    assert out.count("(₹ in crore)") == 2, out
    assert india.clean_text("(In ` crore, except per equity share data)") == "(₹ in crore, except per equity share data)"
    assert india.clean_text("(in crore)") == "(in crore)"                    # no glyph: unchanged
    assert india.clean_text("Section I of the Act; Schedule H applies") == "Section I of the Act; Schedule H applies"
    assert india.clean_text("## footnote marker") == "* footnote marker"   # would otherwise read as a heading
    print("PASS: the rupee glyph (I/H/C/`) is restored before crore/lakh amounts and in unit headers only")


def test_sections_kept_and_skipped() -> None:
    pages = [
        page("Acme Limited", "2", "Notice", "Notice is hereby given of the AGM. " + FILLER),
        page("Acme Limited", "3", "Resolutions and e-voting instructions. " + FILLER),       # notice continues
        page("Our story", "Long arc of value creation. " + FILLER),                          # integrated report
        page(HEADER, "16 17", "Management Discussion and Analysis", "Demand was strong. " + FILLER),
        page(HEADER, "18 19", "Outlook for next year. " + FILLER),                           # MD&A continues
        page(HEADER, "40 41", "Business Responsibility and Sustainability Report", "Principle 6. " + FILLER),
        page(HEADER, "42 43", "ESSENTIAL INDICATORS", "Energy use. " + FILLER),
        page(HEADER, "90 91", "Standalone Financial Statements", "Standalone revenue ₹ 100 crore. " + FILLER),
        page(HEADER, "92 93", "STANDALONE FINANCIALS", "Parent-only notes. " + FILLER),
        page(HEADER, "120 121", "Consolidated Financial Statements", "Consolidated revenue ₹ 150 crore. " + FILLER),
        page("Notes", FILLER, "Segment information. " + FILLER,
             "to the Consolidated Financial Statements for the year ended 31st March 2026 (Contd.)"),
        page("Notes", FILLER, "to the Standalone Financial Statements for the year ended 31st March 2026 (Contd.)"),
    ]
    text, stats = india.build_document(pages)
    headings = [l for l in text.split("\n") if l.startswith("## ")]
    assert [h.split(" (")[0] for h in headings] == [
        "## Integrated Report", "## Management Discussion and Analysis", "## Management Discussion and Analysis",
        "## Consolidated Financial Statements", "## Consolidated Financial Statements"], headings
    assert headings[1] == "## Management Discussion and Analysis (p. 16–17)"
    assert "Standalone revenue" not in text and "Parent-only" not in text and "AGM" not in text
    assert "Principle 6" not in text and "Energy use" not in text
    assert "Long arc of value creation" in text and "Consolidated revenue ₹ 150 crore" in text
    assert HEADER not in text  # running headers are stripped from the body
    assert stats[("Standalone Financial Statements", False)] == 3 and stats[("Notice", False)] == 2
    print("PASS: notice/BRSR/standalone skipped (header, SBI-style and footer forms); MD&A and consolidated kept with page labels")


def test_slug_and_title() -> None:
    report = {"fromYr": "2025", "toYr": "2026", "fileName": "https://example/ar.pdf", "broadcast_dttm": "28-MAY-2026 13:39:47"}
    assert india.slug("RELIANCE", report) == "tier2_nse_RELIANCE-AR-FY2025-26"
    rows = {"data": [{"fromYr": "2024", "toYr": "2025", "fileName": "a.pdf"}, {"fromYr": "2025", "toYr": "2026", "fileName": "b.pdf"},
                     {"fromYr": "2026", "toYr": "2027", "fileName": ""}]}

    class Resp:
        def raise_for_status(self):
            pass

        def json(self):
            return rows

    asked = []
    latest = india.latest_report("TCS", get=lambda url, **kw: asked.append((url, kw)) or Resp())
    assert latest["fileName"] == "b.pdf"                               # newest year that has a file
    assert "symbol=TCS" in asked[0][0] and "Mozilla" not in asked[0][1]["headers"]["User-Agent"]  # honest UA
    print("PASS: newest report with a file is chosen; requests carry an honest User-Agent")


def test_subsidiary_and_board_pages() -> None:
    parent = "Reliance Industries"
    pages = [
        page(HEADER, "200 201", "Consolidated Financial Statements", "Group revenue ₹ 11,75,919 crore. " + FILLER),
        page("Consolidated Financial StatementsJio\tPlatforms\tLimited", "Reliance Industries Limited288",
             "(I in crore)", "Revenue from Operations 1,46,885 " + FILLER),
        page("Consolidated Financial Statements", "Reliance Retail Ventures Limited", HEADER, "286 287", FILLER),
        page("Consolidated Financial Statements", "Reliance Industries Limited", HEADER, "290 291", FILLER),
        page("Dear Members,", "The Board of Directors hereby submits the report of the business. " + FILLER),
        page("N16", "Item No. 6 – Material Related Party Transaction(s) with", "Tata Capital Limited", FILLER),
    ]
    text, _ = india.build_document(pages, parent=parent)
    headings = [l.split(" (")[0] for l in text.split("\n") if l.startswith("## ")]
    assert headings == ["## Consolidated Financial Statements",
                        "## Consolidated Financial Statements — Subsidiary: Jio Platforms Limited",
                        "## Consolidated Financial Statements — Subsidiary: Reliance Retail Ventures Limited",
                        "## Consolidated Financial Statements",        # the parent's own name is not a subsidiary
                        "## Board's Report"], headings                 # the AGM notice item is skipped
    assert "Item No. 6" not in text
    print("PASS: subsidiaries' statements labelled by name (tabs normalised); parent pages and Board's report recognised")


def test_dual_tables_labelled() -> None:
    lines = ["Particulars", "Standalone Consolidated", "For the year ended", "March 31,", "YoY", "growth",
             "2026 2025 (%) 2026 2025 (%)",
             "Revenue from operations 1,48,819 1,36,592 9.0 1,78,650 1,62,990 9.6",
             "Profit for the year 29, 211 25,568 29,440 26,713",          # stray space inside 29,211; no growth cells
             "Total 1,15,200 77.4% 1,05,712 77.4% 1,41,272 79.1% 1,28,566 78.9%",   # no matching labels: halves only
             "Odd row 1 2 3",                                             # odd count: left as is
             "The Standalone and Consolidated Financial Statements were audited in 2026, 220 days after year end"]
    out = india.label_dual_tables(lines)
    assert out[7] == ("Revenue from operations — Standalone: 2026 1,48,819, 2025 1,36,592, YoY % 9.0; "
                      "Consolidated: 2026 1,78,650, 2025 1,62,990, YoY % 9.6")
    assert out[8] == "Profit for the year — Standalone: 2026 29,211, 2025 25,568; Consolidated: 2026 29,440, 2025 26,713"
    assert out[9] == ("Total — Standalone: 1,15,200 77.4% 1,05,712 77.4%; Consolidated: 1,41,272 79.1% 1,28,566 78.9%")
    assert out[10] == "Odd row 1 2 3" and out[11] == lines[11]              # prose untouched ("2026, 220" too)
    rev = india.label_dual_tables(["Consolidated Standalone", "FY2026 FY2026", "EBITDA 2,07,911 78,085"])
    assert rev[2] == "EBITDA — Consolidated: FY2026 2,07,911; Standalone: FY2026 78,085"   # header order respected
    prose = ["The Standalone and Consolidated", "Financial Statements 12 34"]
    assert india.label_dual_tables(prose) == prose
    print("PASS: Standalone|Consolidated rows labelled per value (any column order); odd rows and prose untouched")


def test_navigation_bar_is_not_a_section() -> None:
    """ONGC prints a bar listing several sections on every page; it must not label the page (Day 24)."""
    bar = "Annexures to the Board’s Report Standalone Financial Statements Consolidated Financial Statements"
    assert india.page_section([bar, "Unified. Digital. Sustainable.", "Note 12: Borrowings"]) is None
    assert india.page_section([bar, "Overview Consolidated Financial Statements"]) == "Consolidated Financial Statements"
    assert india.page_section(["Standalone Financial Statements", "Balance Sheet"]) == "Standalone Financial Statements"
    print("PASS: a header line naming several sections (a navigation bar) doesn't decide the page's section")


def _downloads(Resp, full: bytes, cut: bytes) -> None:
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        served = iter([cut, cut, full])
        dest = india.download_pdf("https://x/a.pdf", Path(d) / "a.pdf", get=lambda url, **kw: Resp(next(served)))
        assert dest.read_bytes() == full                                  # retried until complete
        try:
            india.download_pdf("https://x/b.pdf", Path(d) / "b.pdf", get=lambda url, **kw: Resp(cut), attempts=3)
        except RuntimeError as e:
            assert "truncated PDF" in str(e) and "after 3 tries" in str(e)
        else:
            raise AssertionError("a truncated PDF must not be saved")
        assert not (Path(d) / "b.pdf").exists()                           # nothing cached to be reused later


def test_downloads_never_keep_a_truncated_pdf() -> None:
    """NSE's archive sometimes serves a cut-off copy with a matching Content-Length; and "M&M" must be URL-encoded."""
    class Resp:
        def __init__(self, data):
            self.content, self.headers = data, {"Content-Length": str(len(data))}

        def raise_for_status(self):
            pass

        def json(self):
            return {"data": []}
    full, cut = b"%PDF-1.7\n...pages...\n%%EOF\n", b"%PDF-1.7\n...pag"
    real_sleep, india.time.sleep = india.time.sleep, lambda s: None   # no real back-off waits in a test
    try:
        _downloads(Resp, full, cut)
    finally:
        india.time.sleep = real_sleep
    asked = []
    india.latest_report("M&M", get=lambda url, **kw: asked.append(url) or Resp(b""))
    assert asked[0].endswith("symbol=M%26M"), asked
    print("PASS: truncated PDFs are retried and never cached; symbols like M&M are URL-encoded for NSE")


if __name__ == "__main__":
    test_navigation_bar_is_not_a_section()
    test_downloads_never_keep_a_truncated_pdf()
    test_rupee_symbol_repair()
    test_sections_kept_and_skipped()
    test_slug_and_title()
    test_subsidiary_and_board_pages()
    test_dual_tables_labelled()
    print("\nAll India report tests passed.")
