"""
scrape_wikipedia.py -- Tier 1 explainers from Wikipedia, for the topics in config/corpus_manifest.yaml.

The website can't republish Investopedia's articles; Wikipedia's text is CC BY-SA 4.0, which allows
it with credit (each source card names Wikipedia, links the exact revision and the licence). Saves
data/raw/tier1/tier1_wikipedia_<slug>.txt + .meta.json in the same format as scrape_tier1.py, with
"## " section headings and the reference sections dropped. Then the usual pipeline:
chunk_documents.py -> enrich_metadata.py -> build_vectorstore.py --only-source wikipedia.

    python scripts/scrape_wikipedia.py --check     # resolve every title, print its first sentence, save nothing
    python scripts/scrape_wikipedia.py
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from datetime import date
from pathlib import Path

import requests
import yaml
from bs4 import BeautifulSoup

ROOT = Path(__file__).parent.parent
MANIFEST = ROOT / "config" / "corpus_manifest.yaml"
OUT_DIR = ROOT / "data" / "raw" / "tier1"
API = "https://en.wikipedia.org/w/api.php"
HEADERS = {"User-Agent": "PortIQ-corpus/0.1 (github.com/Dhritisman007/RAG-FINANCE-LLM)"}  # Wikimedia asks for one
DELAY_SEC = 3.0   # 1/s drew HTTP 429 after ~20 requests
LICENSE = "CC BY-SA 4.0"
# Lists of sources, not explanation: cut from the first of these headings on.
END_SECTIONS = re.compile(r"^(see also|references|notes|external links|further reading|bibliography|sources|"
                          r"citations|footnotes|works cited)$", re.I)


def articles() -> dict[str, str]:
    sources = yaml.safe_load(MANIFEST.read_text())["tier1_fundamentals"]["sources"]
    return next(s for s in sources if s["name"] == "wikipedia")["articles"]


def fetch(title: str) -> dict:
    """-> {title, revid, html, disambiguation} for the article `title` redirects to. HTML, not the plain-text
    extract: the extract flattens formulas ("A=P(1+rn)tn" for A = P(1 + r/n)^(nt)); the HTML keeps their LaTeX."""
    params = {"action": "parse", "format": "json", "formatversion": 2, "redirects": 1, "page": title,
              "prop": "text|revid|properties", "disableeditsection": 1, "disabletoc": 1}
    for attempt in range(4):
        resp = requests.get(API, params=params, headers=HEADERS, timeout=30)
        if resp.status_code != 429:
            break
        time.sleep(int(resp.headers.get("Retry-After", "0") or 0) or 30 * (attempt + 1))   # asked to slow down
    resp.raise_for_status()
    data = resp.json()
    if "error" in data:
        raise ValueError(f"no Wikipedia article {title!r}: {data['error'].get('info', '')}")
    page = data["parse"]
    props = page.get("properties", {})
    return {"title": page["title"], "revid": page["revid"], "html": page["text"],
            "disambiguation": "disambiguation" in (props if isinstance(props, dict) else {})}


def _wrapped(text: str) -> bool:
    """'(1+r/n)' yes; '(a)+(b)' no -- the first bracket must close at the very end."""
    if not (text.startswith("(") and text.endswith(")")):
        return False
    depth = 0
    for i, ch in enumerate(text):
        depth += (ch == "(") - (ch == ")")
        if depth == 0:
            return i == len(text) - 1
    return False


def _group(text: str) -> str:
    """Brackets for a fraction part, power or index unless it's one symbol, a number or already bracketed
    (MathML often puts only ")" under a power: (1+i)^t is mo"(" mn"1" … msup(mo")", t))."""
    return text if len(text) <= 1 or re.fullmatch(r"\d+(?:\.\d+)?", text) or _wrapped(text) else f"({text})"


def mathml_text(el) -> str:
    """MathML -> one-line notation: A=P(1+r/n)^(tn). The page's plain text flattens it to "A=P(1+rn)tn"."""
    if isinstance(el, str):
        return el.strip()
    kids = [k for k in el.children if not (isinstance(k, str) and not k.strip())]
    parts = [mathml_text(k) for k in kids]
    tag = el.name
    if tag in ("mi", "mn", "mo", "mtext", "ms"):
        return re.sub("[\u2061-\u2064]", "", el.get_text()).strip()   # invisible function application / times
    if tag == "annotation" or tag == "annotation-xml":
        return ""
    if tag == "mfrac" and len(parts) == 2:
        return f"{_group(parts[0])}/{_group(parts[1])}"
    if tag == "msup" and len(parts) == 2:
        return f"{_group(parts[0])}^{_group(parts[1])}"
    if tag == "msub" and len(parts) == 2:
        return f"{parts[0]}_{_group(parts[1])}"
    if tag in ("msubsup", "munderover") and len(parts) == 3:
        return f"{parts[0]}_{_group(parts[1])}^{_group(parts[2])}"
    if tag == "munder" and len(parts) == 2:
        return f"{parts[0]}_{_group(parts[1])}"
    if tag == "mover" and len(parts) == 2:
        return f"{parts[0]}^{_group(parts[1])}"
    if tag == "msqrt":
        return f"sqrt({''.join(parts)})"
    if tag == "mroot" and len(parts) == 2:
        return f"root{_group(parts[1])}({parts[0]})"
    if tag == "mtr":
        return ", ".join(p for p in parts if p)
    if tag == "mtable":
        return "; ".join(p for p in parts if p)
    return "".join(parts)   # math, mrow, mstyle, semantics, mtd, mpadded, …


def to_markdown(html: str) -> str:
    """Article HTML -> scrape_tier1's format: '## ' headings, paragraphs and list items, formulas as $LaTeX$;
    no infoboxes, tables, captions, footnote markers or reference sections."""
    soup = BeautifulSoup(html, "html.parser")
    for el in soup.select("table, figure, style, sup.reference, .mw-editsection, .navbox, .reflist, .hatnote, "
                          ".thumb, .shortdescription, .mw-empty-elt, .sidebar"):
        el.decompose()
    for el in soup.select("math"):
        el.replace_with(f" {mathml_text(el)} ")
    out = []
    for el in soup.select("h2, h3, h4, p, li, dd"):
        if el.find_parent(["li", "dd"]) and el.name in ("p", "li", "dd"):
            continue   # its text is part of the enclosing list item
        text = re.sub(r"\s+", " ", el.get_text(" ")).strip()
        text = re.sub(r"\s+([.,;:)])", r"\1", re.sub(r"\(\s+", "(", text))
        if not text:
            continue
        if el.name in ("h2", "h3", "h4"):
            if END_SECTIONS.match(text):
                break
            out.append(f"\n## {text}")
        else:
            out.append(text)
    text = "\n".join(out).strip()
    return re.sub(r"(^## .*\n+)(?=## |\Z)", "", text + "\n", flags=re.M).strip()   # headings left with no text


def slugify(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true", help="resolve titles and print first sentences only")
    args = ap.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    failed, seen = 0, {}
    for i, (topic, title) in enumerate(articles().items()):
        if i:
            time.sleep(DELAY_SEC)
        try:
            page = fetch(title)
        except (requests.RequestException, ValueError, KeyError) as e:
            print(f"✗ {topic}: {e}", file=sys.stderr)
            failed += 1
            continue
        if page["title"] in seen:   # enrich_metadata drops every copy of a repeated passage: both would vanish
            print(f"✗ {topic}: same article as {seen[page['title']]!r} ({page['title']}) — list it once",
                  file=sys.stderr)
            failed += 1
            continue
        seen[page["title"]] = topic
        text = to_markdown(page["html"])
        first = re.split(r"(?<=\.)\s", text, maxsplit=1)[0][:150]
        if page["disambiguation"] or len(text) < 500:
            print(f"✗ {topic}: {page['title']!r} is a disambiguation page or too short", file=sys.stderr)
            failed += 1
            continue
        print(f"{'✓' if not args.check else '·'} {topic} -> {page['title']} ({len(text):,} chars): {first}")
        if args.check:
            continue
        url = f"https://en.wikipedia.org/w/index.php?title={page['title'].replace(' ', '_')}&oldid={page['revid']}"
        meta = {"source": "wikipedia", "tier": 1, "doc_type": "explainer", "title": f"{topic} (Wikipedia)",
                "url": url, "date_fetched": date.today().isoformat(), "date_published": None,
                "company_ticker": None, "arxiv_id": None, "char_count": len(text),
                "license": LICENSE, "wikipedia_title": page["title"]}
        stem = OUT_DIR / f"tier1_wikipedia_{slugify(topic)}"
        stem.with_suffix(".txt").write_text(text + "\n", encoding="utf-8")
        stem.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n")
    if failed:
        sys.exit(f"{failed} topic(s) failed — fix the titles in {MANIFEST.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
