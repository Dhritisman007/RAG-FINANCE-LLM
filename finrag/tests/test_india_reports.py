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


if __name__ == "__main__":
    test_rupee_symbol_repair()
    test_sections_kept_and_skipped()
    test_slug_and_title()
    test_subsidiary_and_board_pages()
    print("\nAll India report tests passed.")
