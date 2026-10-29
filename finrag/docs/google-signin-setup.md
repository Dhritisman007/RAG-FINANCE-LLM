# Google sign-in for PortIQ — setup steps

PortIQ's website mode signs people in with Google through Streamlit's built-in
`st.login` (OpenID Connect). You create the sign-in "client" in Google Cloud;
PortIQ only needs its ID and secret. Steps follow Streamlit's Google tutorial
(docs.streamlit.io → Develop → Tutorials → Authentication → Google), checked
29 Oct 2026. Google renames console menus from time to time — if a name below
doesn't match, look for the closest one.

You can do all of this now and test real Google sign-in **on your Mac**, long
before deploying. Only step 6 waits for the website.

---

## 1. A Google Cloud project
1. Open <https://console.cloud.google.com/> and sign in with the Google account
   that will own PortIQ.
2. Project picker (top left) → **New project** → name it `PortIQ` → **Create**.
   Make sure it's selected afterwards.

No billing account is needed for Google sign-in.

## 2. Branding (what users see on Google's sign-in screen)
1. Open the Google Auth Platform: <https://console.cloud.google.com/auth/overview>
   (if asked to "Get started", do so).
2. Left menu → **Branding**.
3. **App name:** `PortIQ`. **User support email:** your e-mail.
   **Developer contact:** your e-mail. Logo: leave empty for now — adding one
   can trigger Google's brand review.
4. **Authorized domains:** leave empty while testing on your Mac.
   *(Website: added in step 6.)*
5. **Save**.

## 3. Audience (who may sign in while testing)
1. Left menu → **Audience**. User type: **External** (anyone with a Google
   account, once published).
2. Status starts as **Testing** — only listed test users can sign in.
   Under **Test users** → **Add users** → add your own Gmail (and anyone helping
   you test) → **Save**.

## 4. The client (the ID and secret PortIQ uses)
1. Left menu → **Clients** → **Create client**.
2. Application type: **Web application**. Name: `PortIQ` (only you see it).
3. Skip "Authorized JavaScript origins".
4. **Authorized redirect URIs** → **Add URI** →
   `http://localhost:8502/oauth2callback`
   (8502 = the port used below for testing website mode locally; it must match
   exactly, including `/oauth2callback`).
5. **Create**. Open the new client and copy the **Client ID** and
   **Client secret** into your password manager. Never paste them into chat,
   e-mail or a file that is committed to git.

Data access / scopes: nothing to add. PortIQ asks only for the basic
`openid profile email` sign-in, which Streamlit requests by default. PortIQ
stores none of it — only a one-way code of the account ID (see the privacy
notice in the app).

## 5. Test real Google sign-in on your Mac
1. Make a cookie secret (any long random string):
   ```bash
   python3 -c "import secrets; print(secrets.token_urlsafe(32))"
   ```
2. Create `finrag/.streamlit/secrets.toml` — it is git-ignored, so it stays on
   your Mac — with your values:
   ```toml
   [auth]
   redirect_uri = "http://localhost:8502/oauth2callback"
   cookie_secret = "<the random string from step 1>"
   client_id = "<Client ID>"
   client_secret = "<Client secret>"
   server_metadata_url = "https://accounts.google.com/.well-known/openid-configuration"
   ```
3. Run PortIQ in website mode (from `finrag/`), with a throw-away data folder:
   ```bash
   PORTIQ_MODE=hosted PORTIQ_DATA_DIR=/tmp/portiq-users streamlit run app.py --server.port 8502
   ```
4. Open <http://localhost:8502> → **Sign in with Google** → pick a test-user
   account → accept the privacy notice. ⚙ Settings should show "Signed in as
   <your e-mail>"; Sign out should bring back the sign-in page.

If Google shows "Access blocked" or "app not verified": the account isn't in
**Test users** (step 3), or the redirect URI doesn't match exactly (step 4).

Keeping `secrets.toml` there is harmless: the normal local app
(`streamlit run app.py`, without `PORTIQ_MODE=hosted`) never asks anyone to sign in.

## 6. When the website exists (Phase 2 — later)
- Clients → your client → add the website's redirect URI, e.g.
  `https://<your-space>.hf.space/oauth2callback` → **Save**. Keep the localhost
  one for testing.
- On the website, `redirect_uri` in the secrets is the website's URI.
  (Spaces keep secrets as environment variables; the deployment will write
  `secrets.toml` from them at start-up — part of the Phase 2 files.)
- Branding → **Authorized domains**: the website's domain. **TODO, unverified:**
  whether Google accepts `hf.space` there or a custom domain you own is needed.
- Audience → **Publish app** when you open the site to everyone. **Unverified:**
  whether Google asks for a review for a basic-sign-in app with no logo — follow
  its prompts if it does.
