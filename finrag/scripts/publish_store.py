"""
publish_store.py -- copy the vector store to a PRIVATE Hugging Face dataset for the website.

Free hosts (Streamlit Community Cloud) start with an empty disk; the hosted app downloads the store
from this dataset on its first start (rag_loop.download_store, PORTIQ_STORE_REPO + HF_TOKEN).
Visitors can read every cited passage in full, so the website's copy keeps only what may be
republished: SEC filings, annual reports, Wikipedia (CC BY-SA, credited on each source card) and the
arXiv papers under an open licence (config/arxiv_licenses.yaml, from scripts/arxiv_licenses.py).
Investopedia's articles and the other papers stay in the local app only. The dataset must still be
private -- it is a copy of the index, not a publication -- so this refuses to upload into a public one.

Only the files the collection uses are sent (the Chroma folder also keeps old segment folders),
copied with SQLite's backup API so a running app can't hand it a half-written database, and the
copy is opened and searched before anything is uploaded.

    hf auth login                      # once, with a token that can write
    python scripts/publish_store.py --repo <hf-user>/portiq-store --dry-run
    python scripts/publish_store.py --repo <hf-user>/portiq-store

Staging needs ~2x the store's size free; --stage-dir puts it on a roomier drive. --keep leaves the
staged copy there (portiq-store/) so it can be evaluated first: PORTIQ_CHROMA_DIR=<it> evaluate.py.
"""
from __future__ import annotations

import argparse
import os
import shutil
import sqlite3
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import arxiv_licenses  # noqa: E402
import rag_loop as r  # noqa: E402


NOT_REPUBLISHABLE_SOURCES = ["investopedia"]   # left out of the website's copy (see the module docstring)


def open_copy(dst: Path):
    import chromadb
    client = chromadb.PersistentClient(path=str(dst), settings=chromadb.Settings(anonymized_telemetry=False))
    return client.get_collection(r.COLLECTION_NAME)


def closed_papers(coll) -> list[str]:
    """Ids of arXiv passages whose paper isn't under an open licence. Stops if a paper's licence was never
    looked up: unknown isn't open, but it more likely means arxiv_licenses.py wasn't run after new papers."""
    licences = arxiv_licenses.load()
    got = coll.get(where={"tier": 3}, include=["metadatas"])
    missing = sorted({arxiv_licenses.base_id(m["arxiv_id"]) for m in got["metadatas"]} - set(licences))
    if missing:
        sys.exit(f"No licence recorded for {len(missing)} papers ({', '.join(missing[:5])}…): "
                 "run scripts/arxiv_licenses.py. Nothing uploaded.")
    return [i for i, m in zip(got["ids"], got["metadatas"])
            if not arxiv_licenses.is_open(licences[arxiv_licenses.base_id(m["arxiv_id"])])]


def stage(src: Path, dst: Path) -> int:
    """Copy the database and the vector segment folders it references into `dst`, then remove what may
    not be republished from the copy. -> passages removed."""
    db = sqlite3.connect(f"file:{src / 'chroma.sqlite3'}?mode=ro", uri=True)
    segments = [row[0] for row in db.execute("SELECT id FROM segments WHERE scope = 'VECTOR'")]
    out = sqlite3.connect(dst / "chroma.sqlite3")
    db.backup(out)
    db.close()
    out.close()
    for seg in segments:
        shutil.copytree(src / seg, dst / seg)
    coll = open_copy(dst)
    before = coll.count()
    coll.delete(where={"source": {"$in": NOT_REPUBLISHABLE_SOURCES}})
    closed = closed_papers(coll)
    for i in range(0, len(closed), 5000):
        coll.delete(ids=closed[i:i + 5000])
    removed = before - coll.count()
    del coll
    out = sqlite3.connect(dst / "chroma.sqlite3")
    out.execute("VACUUM")   # give the deleted articles' pages back: they'd still be in the file otherwise
    out.close()
    return removed


def check(dst: Path, expected: int) -> None:
    coll = open_copy(dst)
    if coll.count() != expected:
        sys.exit(f"Copy has {coll.count()} passages, expected {expected} — not uploading.")
    if coll.get(where={"source": {"$in": NOT_REPUBLISHABLE_SOURCES}}, limit=1)["ids"] or closed_papers(coll):
        sys.exit("The copy still holds passages that may not be republished — not uploading.")
    hits = coll.query(query_embeddings=[r.get_embedder().embed_query("What is a P/E ratio?")], n_results=1)
    if not hits["ids"][0]:
        sys.exit("The copy returned nothing for a test search — not uploading.")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo", required=True, help="<hf-user>/<dataset-name>; created private if missing")
    ap.add_argument("--dry-run", action="store_true", help="stage and check the copy, upload nothing")
    ap.add_argument("--stage-dir", default=None, help="where to build the copy (default: the system temp folder)")
    ap.add_argument("--keep", action="store_true", help="keep the copy in <stage-dir>/portiq-store (replaced if there)")
    args = ap.parse_args()
    if args.keep and not args.stage_dir:
        ap.error("--keep needs --stage-dir")

    live = r.open_collection()
    if live is None:
        sys.exit(1)
    total = live.count()
    with tempfile.TemporaryDirectory(prefix="portiq-store-", dir=args.stage_dir) as tmp:
        dst = Path(tmp) / "portiq-store"
        dst.mkdir()
        os.environ["SQLITE_TMPDIR"] = tmp   # VACUUM writes a second full copy to SQLite's temp folder
        removed = stage(r.CHROMA_DIR, dst)
        expected = total - removed
        files = sorted(p for p in dst.rglob("*") if p.is_file() and not p.name.startswith("._"))
        size = sum(f.stat().st_size for f in files)
        print(f"Staged {len(files)} files, {size / 1e9:.2f} GB: {expected:,} passages "
              f"({removed:,} not republishable, left out)")
        check(dst, expected)
        print("Copy opens and searches correctly; only republishable passages in it.")
        if args.keep:
            kept = Path(args.stage_dir) / "portiq-store"
            shutil.rmtree(kept, ignore_errors=True)
            shutil.move(str(dst), kept)
            dst = kept
            print(f"Kept at {kept}")
        if args.dry_run:
            print("Dry run: nothing uploaded.")
            return

        from huggingface_hub import HfApi
        api = HfApi()
        api.create_repo(args.repo, repo_type="dataset", private=True, exist_ok=True)
        if not api.repo_info(args.repo, repo_type="dataset").private:
            sys.exit(f"{args.repo} is public — make it private on huggingface.co first. Nothing uploaded.")
        api.upload_folder(repo_id=args.repo, repo_type="dataset", folder_path=str(dst),
                          ignore_patterns=["._*", ".DS_Store"],   # macOS metadata on external drives
                          commit_message=f"PortIQ vector store: {expected:,} republishable passages")
        print(f"Uploaded to the private dataset {args.repo}.")


if __name__ == "__main__":
    main()
