Broker fixtures for offline tests. They follow the **official documentation's example
responses** (Upstox: get-holdings, ltp-v3, error codes + SDK Problem model; Dhan: v2
portfolio, market-quote, annexure; Groww: curl/portfolio, live-data, error structure),
extended with a second holding. They are NOT recordings from a live account — no
broker sandbox exposes holdings (Upstox's sandbox covers order APIs only).
Values marked "assumed" in a file's "_note" (e.g. Dhan's errorType text) aren't shown in the docs.
