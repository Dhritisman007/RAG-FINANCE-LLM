# Putting PortIQ online (free): Streamlit Community Cloud

| Part | Service (free tier) | What it holds |
|---|---|---|
| Web app | Streamlit Community Cloud (≤ 2.7 GB memory, sleeps after 12 h idle) | the app, from this GitHub repo |
| Search index | Hugging Face — **private** dataset | the 1.3 GB vector store (`scripts/publish_store.py`) |
| Answers | Groq API (~200k tokens/day per model ≈ 50 questions) | nothing kept by PortIQ; see the privacy notice |
| Users' portfolios | Neon Postgres | encrypted copies (`scripts/remote_store.py`) — the app's disk is wiped on restart |
| Sign-in | Google (OAuth client) | nothing: PortIQ keeps a one-way code of the account ID |

Limits checked 17 Nov 2026 (Streamlit's "Manage your app" page, Groq's rate-limit page); they change.

The website's settings are **secrets** in Streamlit Cloud — the template is
[`streamlit-secrets.example.toml`](streamlit-secrets.example.toml). Keep the filled-in copy in your
password manager or a file **outside** this repo. Never put it in `finrag/.streamlit/secrets.toml`: local
runs read that file, and `PORTIQ_MODE = "hosted"` there would turn your local app into website mode.

## 1. Hugging Face: the search index
1. Sign up at <https://huggingface.co/join>.
2. Settings → Access Tokens → **Create new token** → type **Write** (for the upload from your Mac).
   On the Mac: `hf auth login` and paste it (stored in `~/.cache/huggingface`, not in the repo).
3. Upload (checks the copy first; creates the dataset **private** and refuses a public one):
   ```bash
   python scripts/publish_store.py --repo <hf-user>/portiq-store --stage-dir "/Volumes/Crucial X9/finrag_staging"
   ```
4. Create a second token, type **Read** — this one goes into the website's secrets (`HF_TOKEN`).

## 2. Groq: the model that writes answers
1. Sign up at <https://console.groq.com> → API Keys → **Create API key**.
2. For the evaluation on your Mac, add to `finrag/.env` (git-ignored):
   `PORTIQ_LLM_API_KEY=gsk_...` — the model is chosen after `evaluate.py` compares the candidates.
3. Website secrets: `PORTIQ_LLM_URL`, `PORTIQ_LLM_MODEL`, `PORTIQ_LLM_API_KEY`, `PORTIQ_ALLOW_REMOTE_LLM = "1"`.

## 3. Neon: users' saved portfolios
1. Sign up at <https://neon.tech> → New project `portiq`. **TODO, unverified:** which region Streamlit Cloud's
   servers are in; the region only affects how fast a portfolio saves and loads.
2. Dashboard → **Connect** → copy the connection string (`postgresql://…?sslmode=require`) → `PORTIQ_DATABASE_URL`.
3. Encryption key → `PORTIQ_DATA_KEY` (losing it makes the saved portfolios unreadable — keep a copy):
   ```bash
   python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"
   ```

## 4. Streamlit Community Cloud: the website
1. <https://share.streamlit.io> → **Continue with GitHub** (the account that owns `RAG-FINANCE-LLM`).
2. **Create app** → *Deploy a public app from GitHub*: repository `Dhritisman007/RAG-FINANCE-LLM`, branch `main`,
   main file `finrag/app.py`, App URL e.g. `portiq` → `https://portiq.streamlit.app` (if taken, pick another and
   use it below).
3. **Advanced settings** → Python **3.13** → paste the secrets (without `[auth]` for now — next step) → **Deploy**.
4. First start: installing takes a few minutes; then it downloads the index (1.3 GB) and the embedding model.
   Without `[auth]` the page says "Sign-in isn't set up" — expected.

## 5. Google sign-in
Follow [`google-signin-setup.md`](google-signin-setup.md) steps 1–4, but in step 4 add the redirect URI
`https://<your-app>.streamlit.app/oauth2callback` (keep the localhost one for testing). Then in Streamlit Cloud →
your app → ⋮ → **Settings → Secrets**, add the `[auth]` section (cookie secret:
`python -c "import secrets; print(secrets.token_urlsafe(32))"`) → **Save** (the app restarts).

While the Google app is in **Testing**, only the test users you list can sign in (Audience → Test users).
**TODO, unverified:** what Google asks before you **Publish app** for everyone.

## 6. Check
- Sign in → privacy notice names Streamlit Community Cloud, `api.groq.com` and Neon.
- Ask a question; add a holding; ⚙ Settings → **Reboot app** in Streamlit Cloud → sign in again: the holding is
  still there (restored from Neon).
- Apps → ⋮ → **Logs** shows any error (no secrets are printed).
