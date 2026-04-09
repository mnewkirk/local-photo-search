# Google Photos Upload — Setup Guide

This is a one-time setup. After completing it, you can upload collections and individual photos to your Google Photos library directly from the Photo Search UI.

---

## Step 1 — Create a Google Cloud Project

1. Go to [console.cloud.google.com](https://console.cloud.google.com)
2. Click the project dropdown at the top → **New Project**
3. Give it a name (e.g. "Photo Search") and click **Create**
4. Make sure the new project is selected in the dropdown before continuing

## Step 2 — Enable the Photos Library API

1. In the left sidebar go to **APIs & Services → Library**
2. Search for **Photos Library API**
3. Click it and hit **Enable**

## Step 3 — Configure the OAuth Consent Screen

1. Go to **APIs & Services → OAuth consent screen**
2. Under "User Type" choose **External**, then click **Create**
3. Fill in the required fields:
   - **App name**: Photo Search (or anything you like)
   - **User support email**: your Google email address
   - **Developer contact information**: your email address
4. Click **Save and Continue** through the remaining steps (Scopes, Test users, Summary) — you don't need to add any scopes or test users here
5. Click **Back to Dashboard**

## Step 4 — Create OAuth 2.0 Credentials

> **Important**: Choose **Desktop app**, not "Web application". Desktop app credentials allow `http://localhost` automatically — no redirect URI registration required, and no "Authorized JavaScript origins" or "Authorized redirect URIs" to fill in.

1. Go to **APIs & Services → Credentials**
2. Click **Create Credentials → OAuth client ID**
3. Application type: **Desktop app**
4. Name: anything (e.g. "Photo Search")
5. Click **Create**
6. In the confirmation dialog, click **Download JSON**

## Step 5 — Place the credentials file

Rename the downloaded file to:
```
google_client_secret.json
```

Place it in the same folder as `photo_index.db`.

**On the NAS:** copy it into the Docker volume that contains `photo_index.db`:
```bash
scp google_client_secret.json user@<nas-ip>:/path/to/photosearch/
```

Or point to it via environment variable instead:
```bash
GOOGLE_CLIENT_SECRET_FILE=/full/path/to/google_client_secret.json
```

## Step 6 — Connect from Photo Search

1. Open Photo Search in your browser
2. Go to **Collections** and open any collection
3. Click **🖼️ Upload to Google Photos**
4. Click **Connect Google Photos** — a sign-in window will open
5. Sign in with your Google account and click **Allow**

### If the sign-in window shows an error page

This happens when you're accessing Photo Search on a NAS from a different machine (e.g. Mac → NAS). Google redirects to `localhost` which doesn't reach the NAS.

**Fix**: Look at the URL of the error page. It will look like:
```
http://localhost:8000/api/google/callback?code=4/0AX4XfWi...&scope=...
```

Copy the value after `code=` (up to the next `&`) and paste it into the **"Got an error page?"** field in the connect dialog, then click **Submit code**.

---

## For NAS deployment (Docker)

Add the `GOOGLE_CLIENT_SECRET_FILE` env var in `docker-compose.nas.yml` if the file is outside the default location:

```yaml
environment:
  - GOOGLE_CLIENT_SECRET_FILE=/data/google_client_secret.json
```

No `GOOGLE_REDIRECT_URI` setting is needed — the redirect URI is handled automatically.

---

## Disconnect

To revoke access and clear stored tokens:
```bash
curl -X DELETE http://<server>:8000/api/google/disconnect
```

---

## Troubleshooting

**"Setup required" in the UI** — `google_client_secret.json` isn't found. Check the file name and location, then restart the server.

**"Error 400: redirect_uri_mismatch"** — You used Web application credentials instead of Desktop app. Re-create the credentials as Desktop app type and re-download the JSON.

**"Access blocked: This app's request is invalid"** — Google is blocking the app because the OAuth consent screen is in testing mode. In Google Cloud Console, go to **OAuth consent screen** and click **Publish App** (it's fine for personal use — just confirms you've reviewed it). Alternatively, add your email under **Test users**.

**Code exchange fails with "invalid_grant"** — The authorization code has already been used or expired (they're single-use and expire in a few minutes). Click **Re-open sign-in** and try again.

**Tokens expire** — Access tokens auto-refresh. If you get auth errors after a long time, click Connect again to re-authorize.
