"""
publish_store.py -- copy the vector store to a PRIVATE Hugging Face dataset for the website.

Free hosts (Streamlit Community Cloud) start with an empty disk; the hosted app downloads the store
from this dataset on its first start (rag_loop.download_store, PORTIQ_STORE_REPO + HF_TOKEN).
The store holds web articles that aren't ours to republish, so the dataset must stay private:
this refuses to upload into a public one.

Only the files the collection uses are sent (the Chroma folder also keeps old segment folders),
copied with SQLite's backup API so a running app can't hand it a half-written database, and the
copy is opened and searched before anything is uploaded.

    hf auth login                      # once, with a token that can write
    python scripts/publish_store.py --repo <hf-user>/portiq-store --dry-run
    python scripts/publish_store.py --repo <hf-user>/portiq-store

Staging needs ~2x the store's size free; --stage-dir puts it on a roomier drive.
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
import rag_loop as r  # noqa: E402


def stage(src: Path, dst: Path) -> list[Path]:
    """Copy the database and the vector segment folders it references into `dst`."""
    db = sqlite3.connect(f"file:{src / 'chroma.sqlite3'}?mode=ro", uri=True)
    segments = [row[0] for row in db.execute("SELECT id FROM segments WHERE scope = 'VECTOR'")]
    out = sqlite3.connect(dst / "chroma.sqlite3")
    db.backup(out)
    db.close()
    out.execute("VACUUM")
    out.close()
    for seg in segments:
        shutil.copytree(src / seg, dst / seg)
    return sorted(p for p in dst.rglob("*") if p.is_file() and not p.name.startswith("._"))


def check(dst: Path, expected: int) -> None:
    import chromadb
    client = chromadb.PersistentClient(path=str(dst), settings=chromadb.Settings(anonymized_telemetry=False))
    coll = client.get_collection(r.COLLECTION_NAME)
    if coll.count() != expected:
        sys.exit(f"Copy has {coll.count()} passages, the store has {expected} — not uploading.")
    hits = coll.query(query_embeddings=[r.get_embedder().embed_query("What is a P/E ratio?")], n_results=1)
    if not hits["ids"][0]:
        sys.exit("The copy returned nothing for a test search — not uploading.")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo", required=True, help="<hf-user>/<dataset-name>; created private if missing")
    ap.add_argument("--dry-run", action="store_true", help="stage and check the copy, upload nothing")
    ap.add_argument("--stage-dir", default=None, help="where to build the copy (default: the system temp folder)")
    args = ap.parse_args()

    live = r.open_collection()
    if live is None:
        sys.exit(1)
    expected = live.count()
    with tempfile.TemporaryDirectory(prefix="portiq-store-", dir=args.stage_dir) as tmp:
        dst = Path(tmp)
        os.environ["SQLITE_TMPDIR"] = tmp   # VACUUM writes a second full copy to SQLite's temp folder
        files = stage(r.CHROMA_DIR, dst)
        size = sum(f.stat().st_size for f in files)
        print(f"Staged {len(files)} files, {size / 1e9:.2f} GB ({expected:,} passages)")
        check(dst, expected)
        print("Copy opens and searches correctly.")
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
                          commit_message=f"PortIQ vector store: {expected:,} passages")
        print(f"Uploaded to the private dataset {args.repo}.")


if __name__ == "__main__":
    main()
