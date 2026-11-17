"""
Website deployment tests (Streamlit Community Cloud): settings arrive before anything reads them,
the vector store download is all-or-nothing, and the website's config can't start a shared
local-mode app. Offline.

    python tests/test_hosting.py
"""
import os
import re
import sys
import tempfile
import tomllib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

import huggingface_hub  # noqa: E402
import huggingface_hub.constants as hub_constants  # noqa: E402
import rag_loop  # noqa: E402
import workspace  # noqa: E402

ROOT = Path(__file__).parent.parent
SECRETS_EXAMPLE = ROOT / "docs" / "streamlit-secrets.example.toml"


def test_secrets_become_settings() -> None:
    keys = ("PORTIQ_T_MODE", "PORTIQ_T_LIMIT", "PORTIQ_T_FLAG", "PORTIQ_T_SET")
    old = {k: os.environ.pop(k, None) for k in keys}
    os.environ["PORTIQ_T_SET"] = "from-env"
    try:
        workspace.settings_from_secrets({"PORTIQ_T_MODE": "hosted", "PORTIQ_T_LIMIT": 50, "PORTIQ_T_FLAG": True,
                                         "PORTIQ_T_SET": "from-secrets", "auth": {"client_id": "x"}})
        assert os.environ["PORTIQ_T_MODE"] == "hosted" and os.environ["PORTIQ_T_LIMIT"] == "50"
        assert "PORTIQ_T_FLAG" not in os.environ          # a bool isn't a setting this app reads
        assert os.environ["PORTIQ_T_SET"] == "from-env"   # a real environment variable wins
        assert "auth" not in os.environ                   # tables stay in st.secrets (st.login reads [auth])
    finally:
        for k, v in old.items():
            os.environ.pop(k, None)
            if v is not None:
                os.environ[k] = v
    print("PASS (a): top-level secrets become environment settings; env wins, tables and bools stay out")


def test_store_download_is_all_or_nothing() -> None:
    real_download, real_dir = huggingface_hub.snapshot_download, rag_loop.CHROMA_DIR
    calls = []

    def fake_download(repo_id, repo_type, local_dir):
        calls.append((repo_id, repo_type, hub_constants.HF_HUB_OFFLINE))
        Path(local_dir).mkdir(parents=True, exist_ok=True)   # a retry resumes into the .partial folder
        (Path(local_dir) / "chroma.sqlite3").write_bytes(b"db")
        if repo_id.endswith("broken"):
            raise ConnectionError("cut off")
        return str(local_dir)

    with tempfile.TemporaryDirectory() as tmp:
        rag_loop.CHROMA_DIR = Path(tmp) / "store"
        huggingface_hub.snapshot_download = fake_download
        was_offline, hub_constants.HF_HUB_OFFLINE = hub_constants.HF_HUB_OFFLINE, True  # as after get_embedder()
        try:
            try:
                rag_loop.download_store("me/broken")
                raise AssertionError("a failed download must raise")
            except ConnectionError:
                pass
            assert not rag_loop.CHROMA_DIR.exists()           # a cut-off store is never opened
            assert hub_constants.HF_HUB_OFFLINE is True        # offline mode restored after the download
            rag_loop.download_store("me/portiq-store")
            assert (rag_loop.CHROMA_DIR / "chroma.sqlite3").read_bytes() == b"db"
            assert calls == [("me/broken", "dataset", False), ("me/portiq-store", "dataset", False)]
        finally:
            huggingface_hub.snapshot_download, rag_loop.CHROMA_DIR = real_download, real_dir
            hub_constants.HF_HUB_OFFLINE = was_offline
    print("PASS (b): the store downloads online even after the embedder went offline, and only a complete one is kept")


def test_website_config_cannot_start_a_shared_local_app() -> None:
    cfg = tomllib.loads((ROOT.parent / ".streamlit" / "config.toml").read_text())
    assert cfg["browser"]["gatherUsageStats"] is False
    # No server.address: without PORTIQ_MODE=hosted the app is in local mode, isn't bound to 127.0.0.1,
    # and refuses to start (app.py's privacy guard) instead of showing everyone one portfolio.
    assert "address" not in cfg.get("server", {})
    print("PASS (c): the website config sends no usage statistics and doesn't bind 127.0.0.1")


def test_secrets_example_names_only_real_settings() -> None:
    example = tomllib.loads(SECRETS_EXAMPLE.read_text())
    sources = [ROOT / "app.py", *sorted(p for p in (ROOT / "scripts").rglob("*.py") if not p.name.startswith("._"))]
    code = "\n".join(p.read_text() for p in sources)   # ._* = macOS metadata on the external drive
    read = set(re.findall(r"""environ(?:\.get)?[\[(]\s*["']([A-Z_]+)["']""", code)) | {"HF_TOKEN"}  # HF_TOKEN: huggingface_hub
    unknown = [k for k, v in example.items() if not isinstance(v, dict) and k not in read]
    assert not unknown, f"secrets example names settings no code reads (typo?): {unknown}"
    assert set(example["auth"]) >= {"redirect_uri", "cookie_secret", "client_id", "client_secret", "server_metadata_url"}
    assert re.search(r"^secrets\.toml$", (ROOT.parent / ".gitignore").read_text(), re.M)
    print("PASS (d): every setting in the secrets example is read by the code; secrets.toml is git-ignored")


if __name__ == "__main__":
    test_secrets_become_settings()
    test_store_download_is_all_or_nothing()
    test_website_config_cannot_start_a_shared_local_app()
    test_secrets_example_names_only_real_settings()
    print("\nAll hosting tests passed.")
