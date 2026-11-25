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

from bs4 import BeautifulSoup

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

import arxiv_licenses  # noqa: E402
import huggingface_hub  # noqa: E402
import huggingface_hub.constants as hub_constants  # noqa: E402
import publish_store  # noqa: E402
import rag_loop  # noqa: E402
import scrape_wikipedia  # noqa: E402
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


def test_only_open_papers_reach_the_website() -> None:
    assert arxiv_licenses.is_open("http://creativecommons.org/licenses/by/4.0/")
    assert arxiv_licenses.is_open("http://creativecommons.org/licenses/by-sa/4.0/")
    assert arxiv_licenses.is_open("http://creativecommons.org/publicdomain/zero/1.0/")
    for closed in ("http://creativecommons.org/licenses/by-nc-nd/4.0/", "http://creativecommons.org/licenses/by-nc-sa/4.0/",
                   "http://arxiv.org/licenses/nonexclusive-distrib/1.0/", None, ""):
        assert not arxiv_licenses.is_open(closed), closed
    assert arxiv_licenses.base_id("0802.1747v1") == "0802.1747" and arxiv_licenses.base_id("math/0409076v2") == "math/0409076"

    class FakeCollection:
        def __init__(self, rows): self.rows = rows
        def get(self, where, include):
            return {"ids": [i for i, _ in self.rows], "metadatas": [m for _, m in self.rows]}

    real_load = arxiv_licenses.load
    arxiv_licenses.load = lambda: {"2601.17247": "http://creativecommons.org/licenses/by/4.0/",
                                   "2507.02027": "http://arxiv.org/licenses/nonexclusive-distrib/1.0/"}
    try:
        rows = [("a_0", {"arxiv_id": "2601.17247v2"}), ("b_0", {"arxiv_id": "2507.02027v1"}), ("b_1", {"arxiv_id": "2507.02027v1"})]
        assert publish_store.closed_papers(FakeCollection(rows)) == ["b_0", "b_1"]
        try:   # a paper whose licence was never looked up stops the upload
            publish_store.closed_papers(FakeCollection(rows + [("c_0", {"arxiv_id": "2612.00001v1"})]))
            raise AssertionError("an unknown licence must stop the upload")
        except SystemExit as e:
            assert "arxiv_licenses.py" in str(e)
    finally:
        arxiv_licenses.load = real_load
    assert publish_store.NOT_REPUBLISHABLE_SOURCES == ["investopedia"]
    print("PASS (e): only CC BY / BY-SA / CC0 papers go on the website; an unlooked-up licence stops the upload")


def test_wikipedia_article_to_text() -> None:
    soup = BeautifulSoup('<math><mrow><mi>A</mi><mo>=</mo><mi>P</mi><msup><mrow><mo>(</mo><mn>1</mn><mo>+</mo>'
                         '<mfrac><mi>r</mi><mi>n</mi></mfrac><mo>)</mo></mrow><mrow><mi>t</mi><mi>n</mi></mrow></msup>'
                         '</mrow></math>', "html.parser")
    assert scrape_wikipedia.mathml_text(soup.math) == "A=P(1+r/n)^(tn)"   # the plain-text extract gives "A=P(1+rn)tn"
    soup = BeautifulSoup('<math><mfrac><msub><mi>R</mi><mi>t</mi></msub><mrow><mo>(</mo><mn>1</mn><mo>+</mo><mi>i</mi>'
                         '<msup><mo>)</mo><mi>t</mi></msup></mrow></mfrac></math>', "html.parser")
    assert scrape_wikipedia.mathml_text(soup.math) == "(R_t)/((1+i)^t)"    # MathML puts only ")" under the power
    html = ('<div><table class="infobox"><tr><td>Infobox</td></tr></table><p>Compound interest is interest on interest.'
            '<sup class="reference">[1]</sup></p><div class="mw-heading"><h2>Formula</h2></div>'
            '<p>The amount is <math class="mwe-math-element"><mi>A</mi></math> after one year.</p>'
            '<ul><li>P is the principal</li></ul><h2>Empty</h2><h2>See also</h2><ul><li>Simple interest</li></ul>'
            '<h2>References</h2><p>Cited work</p></div>')
    assert scrape_wikipedia.to_markdown(html) == ("Compound interest is interest on interest.\n\n## Formula\n"
                                                  "The amount is A after one year.\nP is the principal")
    print("PASS (f): Wikipedia HTML -> '## ' sections, readable formulas; no infobox, footnotes or reference sections")


if __name__ == "__main__":
    test_secrets_become_settings()
    test_store_download_is_all_or_nothing()
    test_website_config_cannot_start_a_shared_local_app()
    test_secrets_example_names_only_real_settings()
    test_only_open_papers_reach_the_website()
    test_wikipedia_article_to_text()
    print("\nAll hosting tests passed.")
