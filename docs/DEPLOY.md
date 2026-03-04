# LegacyLens Deployment Guide

Steps to deploy LegacyLens for production use. The app runs as a container (Docker) and can be deployed to **Google Cloud Run** or any host that runs Docker.

**Note:** The production Docker image uses `index.production.html`, which excludes the Admin UI (Sign In, Corpus, Reingest). Admin is only available when running locally with the dev frontend.

---

## Prerequisites

- **Docker** installed locally
- **Google Cloud** account (for Cloud Run + Vertex AI)
- **Qdrant Cloud** account (Cloud Run is stateless; you cannot use local `.qdrant_data`)

---

## Option A: Deploy to Google Cloud Run

### 1. Set up Qdrant Cloud

1. Create a free cluster at [cloud.qdrant.io](https://cloud.qdrant.io)
2. Create a collection named `legacylens-chunks` (or use the default)
3. Note your cluster URL and API key

### 2. Ingest your codebase into Qdrant Cloud (one-time)

Run ingestion **locally** (or from a VM) so your vectors live in Qdrant Cloud:

```powershell
# From repo root, with .env configured:
$env:QDRANT_URL = "https://your-cluster.qdrant.io"
$env:QDRANT_API_KEY = "your-api-key"
$env:QDRANT_COLLECTION = "legacylens-chunks"
$env:GOOGLE_CLOUD_PROJECT = "your-gcp-project"
$env:GOOGLE_APPLICATION_CREDENTIALS = "path\to\your-key.json"

python -m backend.ingestion.pipeline run --code-root gnucobol-3.2_win
```

### 3. Enable required GCP APIs

```powershell
gcloud services enable run.googleapis.com artifactregistry.googleapis.com
```

### 4. Create Artifact Registry

```powershell
gcloud artifacts repositories create legacylens --repository-format=docker --location=us-central1
```

### 5. Build and push the image

```powershell
# From repo root
docker build -t us-central1-docker.pkg.dev/YOUR_PROJECT_ID/legacylens/legacylens:latest .

# Configure Docker for Artifact Registry
gcloud auth configure-docker us-central1-docker.pkg.dev

# Push
docker push us-central1-docker.pkg.dev/YOUR_PROJECT_ID/legacylens/legacylens:latest
```

Replace `YOUR_PROJECT_ID` with your GCP project ID (e.g. `legacylens-489112`).

**If "Container failed to start":** Check Cloud Run logs for the crash:
```powershell
gcloud logging read 'resource.type="cloud_run_revision" resource.labels.service_name="legacylens" severity>=ERROR' --limit=20 --format="table(timestamp,textPayload)"
```
Test the image locally first:
```powershell
docker run -p 8080:8080 -e QDRANT_URL=... -e QDRANT_API_KEY=... -e GOOGLE_CLOUD_PROJECT=legacylens-489112 -e LLM_ENABLED=true legacylens:latest
```

**If "Image not found" on deploy:** The push may have failed. Run these manually to fix:
```powershell
gcloud auth login
gcloud auth configure-docker us-central1-docker.pkg.dev
docker push us-central1-docker.pkg.dev/YOUR_PROJECT_ID/legacylens/legacylens:latest
```

### 6. Deploy to Cloud Run

```powershell
gcloud run deploy legacylens `
  --image us-central1-docker.pkg.dev/YOUR_PROJECT_ID/legacylens/legacylens:latest `
  --region us-central1 `
  --platform managed `
  --allow-unauthenticated `
  --set-env-vars "QDRANT_URL=https://your-cluster.qdrant.io"
```

**Set secrets via Cloud Run:**

```powershell
# Create secrets first (or use existing)
gcloud secrets create qdrant-api-key --data-file=-
# Paste your Qdrant API key, then Ctrl+Z (Windows) or Ctrl+D (Unix)

# Deploy with secrets
gcloud run deploy legacylens `
  --image us-central1-docker.pkg.dev/YOUR_PROJECT_ID/legacylens/legacylens:latest `
  --region us-central1 `
  --platform managed `
  --allow-unauthenticated `
  --set-env-vars "QDRANT_URL=https://your-cluster.qdrant.io,QDRANT_COLLECTION=legacylens-chunks,GOOGLE_CLOUD_PROJECT=YOUR_PROJECT_ID,LLM_MODEL=gemini-2.0-flash,LLM_ENABLED=true" `
  --set-secrets "QDRANT_API_KEY=qdrant-api-key:latest,ADMIN_TOKEN=admin-token:latest"
```

### 7. Get the URL

After deployment, Cloud Run prints the service URL, e.g.:

```
https://legacylens-xxxxx-uc.a.run.app
```

Open that URL in a browser to use the app.

### Troubleshooting: "0 chunks used"

If chat returns "Answered. 0 chunks used" with no citations:

1. **Check collection has data:** `https://YOUR-URL/health/db` — if `point_count` is 0, the collection is empty.
2. **Ingest into Qdrant Cloud:** Run ingestion locally with `QDRANT_URL` and `QDRANT_API_KEY` set (see step 2 above). Cloud Run cannot use local `.qdrant_data`.
3. **Verify collection name:** Ensure `QDRANT_COLLECTION=legacylens-chunks` matches the collection you ingested into.

---

## Option B: Run on a VM or local server (Docker)

If you prefer to keep using **local Qdrant** (`.qdrant_data`), run the container on a machine with persistent storage:

### 1. Build the image

```powershell
docker build -t legacylens:latest .
```

### 2. Run with local data

```powershell
docker run -p 8080:8080 `
  -v "$(pwd)/.qdrant_data:/app/.qdrant_data" `
  -e GOOGLE_CLOUD_PROJECT=your-project `
  -e GOOGLE_APPLICATION_CREDENTIALS=/app/keys/gcp-key.json `
  -e LLM_MODEL=gemini-2.0-flash `
  -e LLM_ENABLED=true `
  -e ADMIN_TOKEN=your-secure-token `
  -v "$(pwd)/legacylens-gcp-key.json:/app/keys/gcp-key.json:ro" `
  legacylens:latest
```

**Note:** Ingestion must be run on the same host first (or copy `.qdrant_data` into the container).

---

## Environment variables reference

| Variable | Required | Description |
|----------|----------|-------------|
| `QDRANT_URL` | For Cloud Run | Qdrant Cloud cluster URL |
| `QDRANT_API_KEY` | For Cloud Run | Qdrant Cloud API key |
| `QDRANT_COLLECTION` | No | Default: `legacylens-chunks` |
| `GOOGLE_CLOUD_PROJECT` | Yes (for embeddings/LLM) | GCP project ID |
| `LLM_MODEL` | No | e.g. `gemini-2.0-flash` |
| `LLM_ENABLED` | No | `true` to enable chat |
| `ADMIN_TOKEN` | Recommended | Token for admin endpoints |

For Cloud Run, `GOOGLE_APPLICATION_CREDENTIALS` is not needed—the service account is used automatically.

---

## Quick checklist

| Step | Command / Action |
|------|------------------|
| 1 | Create Qdrant Cloud cluster, get URL + API key |
| 2 | Ingest codebase: `python -m backend.ingestion.pipeline run --code-root ...` |
| 3 | `docker build -t legacylens:latest .` |
| 4 | Push to Artifact Registry (if using Cloud Run) |
| 5 | `gcloud run deploy legacylens ...` |
| 6 | Open the Cloud Run URL |
