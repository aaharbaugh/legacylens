# LegacyLens – deploy to Google Cloud Run
# Run from repo root. Requires: gcloud CLI, Docker, Qdrant Cloud + GCP configured.
#
# Before first deploy: ingest your codebase into Qdrant Cloud (run locally with QDRANT_* in .env):
#   python -m backend.ingestion.pipeline run --code-root gnucobol-3.2_win

$ErrorActionPreference = "Stop"
$PROJECT_ID = "legacylens-489112"
$REGION = "us-central1"
$IMAGE = "us-central1-docker.pkg.dev/$PROJECT_ID/legacylens/legacylens:latest"

Write-Host "=== LegacyLens Cloud Run Deployment ===" -ForegroundColor Cyan
Write-Host "Project: $PROJECT_ID" -ForegroundColor Gray

# 1. Enable APIs (idempotent)
Write-Host "`n[1/6] Enabling GCP APIs..." -ForegroundColor Yellow
gcloud services enable run.googleapis.com artifactregistry.googleapis.com --project=$PROJECT_ID

# 2. Create Artifact Registry repo (idempotent – may already exist)
Write-Host "`n[2/6] Ensuring Artifact Registry repo..." -ForegroundColor Yellow
$ErrorActionPreference = "Continue"
gcloud artifacts repositories create legacylens --repository-format=docker --location=$REGION --project=$PROJECT_ID 2>&1 | Out-Null
$ErrorActionPreference = "Stop"
if ($LASTEXITCODE -ne 0) { Write-Host "  (repo may already exist, continuing)" -ForegroundColor Gray }

# 3. Configure Docker for Artifact Registry
Write-Host "`n[3/6] Configuring Docker for Artifact Registry..." -ForegroundColor Yellow
gcloud auth configure-docker us-central1-docker.pkg.dev --quiet

# 4. Build image
Write-Host "`n[4/6] Building Docker image..." -ForegroundColor Yellow
docker build -t $IMAGE .

# 5. Push image
Write-Host "`n[5/6] Pushing image..." -ForegroundColor Yellow
docker push $IMAGE
if ($LASTEXITCODE -ne 0) {
  Write-Host "ERROR: docker push failed. Ensure you're logged in:" -ForegroundColor Red
  Write-Host "  gcloud auth login" -ForegroundColor Gray
  Write-Host "  gcloud auth configure-docker us-central1-docker.pkg.dev" -ForegroundColor Gray
  exit 1
}

# 6. Deploy to Cloud Run
Write-Host "`n[6/6] Deploying to Cloud Run..." -ForegroundColor Yellow
Write-Host "  Set QDRANT_URL, QDRANT_API_KEY, ADMIN_TOKEN via Cloud Run console or --set-env-vars / --set-secrets" -ForegroundColor Gray

# Deploy – reads from .env. Optional: set CLOUD_RUN_SERVICE_ACCOUNT=sa@project.iam.gserviceaccount.com (must have Vertex AI User + you need actAs on it)
$envLines = Get-Content .env -ErrorAction SilentlyContinue
$serviceAccountLine = ($envLines | Where-Object { $_ -match '^CLOUD_RUN_SERVICE_ACCOUNT=' }) | Select-Object -First 1
$SERVICE_ACCOUNT = $null
if ($serviceAccountLine) {
  $SERVICE_ACCOUNT = ($serviceAccountLine -replace '^CLOUD_RUN_SERVICE_ACCOUNT=', '').Trim()
  if (-not $SERVICE_ACCOUNT) { $SERVICE_ACCOUNT = $null }
  else { Write-Host "  Using service account from .env: $SERVICE_ACCOUNT" -ForegroundColor Gray }
}
# If no custom SA, Cloud Run uses default (no --service-account) so deploy works; grant default SA "Vertex AI User" in IAM for embeddings/LLM

$qdrantKey = (($envLines | Where-Object { $_ -match '^QDRANT_API_KEY=' }) -replace '^QDRANT_API_KEY=', '') | Select-Object -First 1
$adminToken = (($envLines | Where-Object { $_ -match '^ADMIN_TOKEN=' }) -replace '^ADMIN_TOKEN=', '') | Select-Object -First 1
$qdrantKey = "$qdrantKey".Trim(); $adminToken = "$adminToken".Trim()
if (-not $qdrantKey -or -not $adminToken) {
  Write-Host "ERROR: .env must contain QDRANT_API_KEY and ADMIN_TOKEN" -ForegroundColor Red
  exit 1
}

$envVars = "QDRANT_URL=https://da99c893-5aa9-4a52-add7-d088a40c4534.us-east4-0.gcp.cloud.qdrant.io,QDRANT_COLLECTION=legacylens-chunks,GOOGLE_CLOUD_PROJECT=$PROJECT_ID,LLM_MODEL=gemini-2.0-flash,LLM_ENABLED=true,QDRANT_API_KEY=$qdrantKey,ADMIN_TOKEN=$adminToken,MIN_VECTOR_SCORE=0,FALLBACK_CHUNK_LINES=120,FALLBACK_OVERLAP_LINES=25,QUERY_TOP_K=50,QUERY_FINAL_K=25"

$deployArgs = @(
  "run", "deploy", "legacylens",
  "--image", $IMAGE,
  "--region", $REGION,
  "--platform", "managed",
  "--allow-unauthenticated",
  "--project", $PROJECT_ID,
  "--memory", "1Gi",
  "--cpu-boost",
  "--set-env-vars", $envVars
)
if ($SERVICE_ACCOUNT) { $deployArgs += "--service-account"; $deployArgs += $SERVICE_ACCOUNT }
gcloud @deployArgs

# Option B (prod): Use Secret Manager instead of env vars for secrets

Write-Host "`n=== Done ===" -ForegroundColor Green
Write-Host "Get your URL: gcloud run services describe legacylens --region $REGION --format='value(status.url)'" -ForegroundColor Gray
