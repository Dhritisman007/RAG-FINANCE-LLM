# Hosted "Connect broker" — design (not implemented)

**Status:** design only. Today FinRAG / PortIQ runs locally and each user brings
their own broker credentials (`.env`, see README → Portfolio sources). This
document describes how a hosted, one-click "Connect broker" button would plug
into the same adapter interface, and what it needs that we don't have.

Claims checked against official docs (Oct 2026) are stated plainly; anything we
could not confirm there is marked **UNVERIFIED**.

---

## 1. How it plugs into the existing adapters

Nothing in `scripts/brokers/` changes shape. The adapters already take their
token as a constructor argument (`get_adapter("upstox", token=...)`) and never
read anything else; locally that token comes from `.env`, hosted it would come
from a per-user token store:

```
browser ──Connect Upstox──▶ /connect/upstox ──302──▶ broker login & consent
browser ◀──302 /callback?code=…&state=…── broker
/callback: verify state → exchange code (server-side, client_secret never leaves the server)
         → encrypt token → token_store.put(user_id, broker, ciphertext, expires_at)
sync worker: token = token_store.get(user_id, broker)  →  get_adapter(broker, token=token)
         → portfolio_sync.sync(...) writing into THAT user's snapshot store
```

- `portfolio_sync.sync()` stays the single entry point; it gains a `user_id`
  (which DB / schema to write) and keeps its guarantees: atomic snapshots, a
  failed sync never wipes the last good one, prices NULL rather than zero,
  `sync_log` per attempt, network limited to the adapter's `allowed_hosts`.
- Answers already carry each snapshot's `source` and `as_of`; that doesn't change.

### Per-broker authorization flows

| Broker | Documented flow | Notes |
|---|---|---|
| **Upstox** | OAuth 2 authorization code: `GET /v2/login/authorization/dialog?client_id&redirect_uri&response_type=code&state` → `code` → `POST /v2/login/authorization/token` (`code`, `client_id`, `client_secret`, `redirect_uri`, `grant_type=authorization_code`). The code is single-use. | Access token valid until **3:30 AM the next day**. The token response also has an `extended_token` "for prolonged usage, primarily for read-only access" — its lifetime is **UNVERIFIED**. No refresh-token grant is documented (**UNVERIFIED** that none exists). |
| **Dhan** | Partner consent flow: `POST auth.dhan.co/partner/generate-consent` (partner_id/secret) → user opens `auth.dhan.co/consent-login?consentId=…` → `tokenId` → `POST …/partner/consume-consent?tokenId=…` → `accessToken`. | Tokens valid **24 hours**; `POST /v2/RenewToken` renews only *active* tokens. Partners must apply via DhanHQ. |
| **Groww** | Only personal flows are documented (token from the Trading APIs page; API key + secret or TOTP via `POST /v1/token/api/access`). | Token "expires daily at 6:00 AM". **No third-party / multi-user flow is documented** — hosted Groww is not possible without one (**UNVERIFIED** that a partner program exists). |

## 2. Per-user encrypted token storage

- Table `broker_tokens(user_id, broker, ciphertext, nonce, key_id, expires_at,
  created_at, last_used_at)` — **never** plaintext, never in logs, never in
  `sync_log`, never returned by any API.
- Envelope encryption: a random data key per user encrypts the token
  (AES-256-GCM); the data key is wrapped by a KMS-held master key. Rotating the
  master key re-wraps data keys without touching ciphertexts.
- Decryption happens only inside the sync worker, immediately before the broker
  call; the plaintext is not cached.
- The existing `test_security.py` secret-leak test becomes a CI gate for the
  hosted code paths too (logs, exception messages, DB rows, error pages).

## 3. Token refresh and expiry

- Store `expires_at` (Upstox: next 03:30 IST; Dhan: +24 h; Groww: next 06:00 IST).
- Before a sync: if expired, skip the broker call, mark the source
  "re-connect needed" (adapter `AuthStatus.EXPIRED`) and keep showing the last
  snapshot with its `as_of` — the same behaviour as `AuthExpired` today.
- Dhan: renew with `RenewToken` while the token is still active.
- Upstox / Groww: daily tokens mean a daily user reconnect unless a longer-lived
  read-only grant is confirmed (Upstox `extended_token`, **UNVERIFIED**).
- An `AuthExpired` from the broker mid-life (revoked by the user at the broker)
  is handled identically: no retry loop, mark "re-connect needed".

## 4. Per-user data isolation

- Every row (snapshots, holdings, sync_log, tokens, chat history) carries
  `user_id`; enforced in the database (row-level security or one schema / one
  encrypted SQLite file per user), not only in application code.
- Retrieval over public documents (annual reports, filings) can be shared; the
  **portfolio context injected into prompts is per user** and assembled only from
  that user's latest snapshots.
- Background jobs run per user with the user id as an explicit parameter; no
  cross-user batching of prompts.
- Hosting breaks today's "nothing leaves the machine" rule by definition: the
  model then runs on a server. Where it runs (our GPU vs a third-party API) and
  what is sent to it must be stated to the user before they connect.

## 5. Revocation and delete-my-data

- **Disconnect broker:** delete the encrypted token; if the broker offers a
  revoke/logout endpoint, call it (Upstox logout endpoint **UNVERIFIED** in the
  pages we read; Groww tokens can be revoked by the user on its Trading APIs
  page; Dhan revocation API **UNVERIFIED**). Tell the user to also revoke the app
  at the broker.
- **Delete my data:** delete tokens, snapshots, holdings, sync_log, chat
  history and derived caches for the user; record a minimal tombstone (user id
  hash, time) in the audit log; confirm by email. Backups expire on a fixed
  schedule, stated in the privacy notice.
- **Export my data:** snapshots + sync history as CSV/JSON.

## 6. Audit logging

Append-only `audit_log(user_id, event, broker, at, ip_hash, result)` for:
connect, consent callback, token stored, token renewed, sync started/finished
(with error *type*, not message), disconnect, data export, data deletion, admin
access. No tokens, no holdings values, no prompt text. Users can see their own
audit trail.

---

## 7. What this requires that we don't have

- **Broker partner approval.**
  - Dhan documents a partner consent flow; partners get `partner_id` /
    `partner_secret` after applying via DhanHQ.
  - Upstox: the API's authorization-code flow is designed for apps acting for
    customers, and Upstox publishes "UpLink Business" multi-client docs; that
    multi-user access needs onboarding/approval is stated on the Upstox
    community forum, not in the API docs — **UNVERIFIED** from official docs.
  - Groww: no documented third-party access — **UNVERIFIED** whether any exists.
- A hosted model server and a privacy design for it (section 4).
- KMS, per-user storage, an auth system for our own users, monitoring.
- A static IP if we wanted Upstox's 1-year read-only Analytics Token for
  portfolio calls (documented: Portfolio APIs need a whitelisted static IP with
  that token) — and it is per account, so it doesn't solve multi-user access.

## 8. Regulatory items to check before hosting for others (open questions — not legal advice)

1. **DPDP Act, 2023 (India)** — holdings are personal (financial) data. Open
   questions: consent notice contents and purpose limitation; data principals'
   rights (access, correction, erasure, grievance) and response timelines;
   breach notification to the Data Protection Board and users; retention limits;
   whether we could be a "Significant Data Fiduciary" (DPO, audits, DPIA);
   processors' obligations if a third-party model API or cloud host processes
   the data; status of the implementing Rules at launch time.
2. **SEBI (Investment Advisers) Regulations, 2013** — could answers about a
   user's own holdings ("risks of *your* RELIANCE position") be read as
   personalised investment advice requiring registration? Where is the line
   between education/information and advice for our UI and disclaimers?
3. **SEBI (Research Analysts) Regulations, 2014** — do company-specific
   summaries of annual reports count as research reports?
4. **Broker API terms** — does each broker's API agreement allow a third party
   to store, process and display client holdings, and to use them with an LLM?
5. **SEBI circulars on API access / algorithmic trading by retail investors** —
   do any obligations reach a read-only portfolio app? (We place no orders.)
6. **Cross-border transfer** — if any processing (hosting, model API) is
   outside India.
7. **Grievance redressal and disclosures** expected of a consumer fintech app.
