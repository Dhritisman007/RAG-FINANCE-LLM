"""
arxiv_licenses.py -- record each indexed arXiv paper's licence (config/arxiv_licenses.yaml).

The website shows cited passages in full, so its copy of the index keeps only papers whose licence
allows republishing (scripts/publish_store.py reads this file). The licence comes from arXiv's own
OAI-PMH record (<license>, a URL); papers without one are under arXiv's default licence, which
gives arXiv -- not us -- the right to distribute them.

    python scripts/arxiv_licenses.py            # fetch missing ones (arXiv asks for 1 request / 3 s)
"""
from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path

import requests
import yaml

ROOT = Path(__file__).parent.parent
RAW = ROOT / "data" / "raw" / "tier3"
OUT = ROOT / "config" / "arxiv_licenses.yaml"
OAI = "https://oaipmh.arxiv.org/oai"
HEADERS = {"User-Agent": "PortIQ-corpus/0.1 (github.com/Dhritisman007/RAG-FINANCE-LLM)"}
DELAY_SEC = 3

# Licences that allow republishing excerpts with credit, commercial or not (CC BY, CC BY-SA, CC0).
OPEN_LICENSE_RE = re.compile(r"creativecommons\.org/(?:licenses/by(?:-sa)?/\d\.\d|publicdomain/zero/1\.0)/?$")


def base_id(arxiv_id: str) -> str:
    """'0802.1747v1' -> '0802.1747' (the licence belongs to the paper, not one version)."""
    return re.sub(r"v\d+$", "", arxiv_id)


def is_open(license_url: str | None) -> bool:
    return bool(license_url and OPEN_LICENSE_RE.search(license_url.strip()))


def load() -> dict[str, str | None]:
    return yaml.safe_load(OUT.read_text()) or {} if OUT.exists() else {}


def fetch_license(paper: str) -> str | None:
    resp = requests.get(OAI, params={"verb": "GetRecord", "identifier": f"oai:arXiv.org:{paper}",
                                     "metadataPrefix": "arXiv"}, headers=HEADERS, timeout=30)
    resp.raise_for_status()
    if "<error" in resp.text:
        raise ValueError(f"arXiv has no record for {paper}")
    m = re.search(r"<license>\s*([^<]+?)\s*</license>", resp.text)
    return m.group(1) if m else None


def main() -> None:
    papers = sorted({base_id(json.loads(p.read_text())["arxiv_id"])
                     for p in RAW.glob("*.meta.json") if not p.name.startswith("._")})
    known = load()
    todo = [p for p in papers if p not in known]
    print(f"{len(papers)} papers, {len(todo)} to look up")
    for i, paper in enumerate(todo):
        if i:
            time.sleep(DELAY_SEC)
        try:
            known[paper] = fetch_license(paper)
        except (requests.RequestException, ValueError) as e:
            print(f"  {paper}: {e} -- left out, run again later", file=sys.stderr)
            continue
        print(f"  {paper}: {known[paper] or '(none: arXiv default licence)'}")
    OUT.write_text("# arXiv paper -> licence URL from arXiv's OAI-PMH record (null = none given: arXiv's default).\n"
                   "# Written by scripts/arxiv_licenses.py; publish_store.py keeps only open ones on the website.\n"
                   + yaml.safe_dump(dict(sorted(known.items())), sort_keys=False))
    n_open = sum(is_open(known.get(p)) for p in papers)
    print(f"Open licence: {n_open} of {len(papers)} papers -> {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
