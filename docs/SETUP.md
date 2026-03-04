# LegacyLens setup – step-by-step

Do these steps in order from the project folder. All commands assume you're in the repo root: `c:\Users\aaron\Documents\GauntletCode\2-legacylens`.

---

## Step 1: Open terminal and go to the project

```powershell
cd c:\Users\aaron\Documents\GauntletCode\2-legacylens
```

---

## Step 2: (Optional) Create a virtual environment

Create the venv:

```powershell
python -m venv .venv
```

Activate it (you must run this so `pip` and `python` use the venv):

```powershell
.\.venv\Scripts\Activate.ps1
```

After activation, your prompt usually shows `(.venv)` at the start. Run the rest of the steps in this same terminal so the venv stays active.

If you get an execution policy error on the activate line, run once: `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser`, then run the activate line again.

---

## Step 3: Install the project

```powershell
pip install -e .
```

**What to expect:** Packages install (qdrant-client, fastapi, uvicorn, etc.). You may see PATH warnings; you can ignore them.

**Check:** No error at the end. You can confirm with:

```powershell
python -c "import backend; print('OK')"
```

---

## Step 4: Ingest the sample COBOL code into the vector DB

```powershell
python -m backend.ingestion.pipeline run --code-root sample_cobol
```

**What to expect:**

- Log line like: `Created collection legacylens-chunks with dim=768`
- Final line: `Done: 1 files, 25 chunks upserted` (or similar numbers)

**Check:** A folder `.qdrant_data` appears in the project (that’s your local vector DB).

If you see “Provide a valid --code-root”, make sure you’re in the repo root and the `sample_cobol` folder exists.

---

## Step 5: Start the API server

In the **same** terminal (with your venv activated and repo as cwd):

```powershell
python -m uvicorn backend.api.main:app --reload --host 0.0.0.0 --port 8000
```

Use `python -m uvicorn` so Windows finds uvicorn from your venv even if the Scripts folder isn’t on PATH.

**What to expect:**

- `Uvicorn running on http://0.0.0.0:8000`
- Leave this terminal open; the server keeps running.

**In your browser, don’t use `http://0.0.0.0:8000`.** Use:

- **http://localhost:8000** or **http://127.0.0.1:8000**

(0.0.0.0 is the address the server *binds* to; you *connect* via localhost.)

---

## Step 6: Run a query

**Option A – New terminal (PowerShell):**

```powershell
cd c:\Users\aaron\Documents\GauntletCode\2-legacylens
.\.venv\Scripts\Activate.ps1   # if you use the venv
```

Then either:

**Option B – Using the query script (easiest):**

```powershell
python scripts/query_cli.py "Where is CALCULATE-INTEREST?"
```

You should see JSON with `query` and `results` (file paths, line numbers, code snippets, scores).

**Option C – Using curl:**

```powershell
curl -X POST http://localhost:8000/query -H "Content-Type: application/json" -d "{\"query\": \"Where is CALCULATE-INTEREST?\"}"
```

**Option D – In the browser:** open **http://localhost:8000/docs** (not 0.0.0.0) and use the “POST /query” endpoint with a body like `{"query": "Where is CALCULATE-INTEREST?", "top_k": 5}`.

---

## Step 7: Try another query

With the server still running:

```powershell
python scripts/query_cli.py "file I/O or customer record"
```

You should get back chunks that mention file handling or customer data.

---

## Summary checklist

| Step | Command | Success looks like |
|------|---------|---------------------|
| 1 | `cd` to repo | You’re in `2-legacylens` |
| 2 | (Optional) `python -m venv .venv` + activate | Prompt shows `(.venv)` |
| 3 | `pip install -e .` | Install finishes without error |
| 4 | `python -m backend.ingestion.pipeline run --code-root sample_cobol` | “Done: 1 files, N chunks upserted” |
| 5 | `python -m uvicorn backend.api.main:app --reload --host 0.0.0.0 --port 8000` | “Uvicorn running on http://0.0.0.0:8000” |
| 6 | `python scripts/query_cli.py "Where is CALCULATE-INTEREST?"` | JSON with `results` and code snippets |

---

## Ingesting a real codebase (e.g. GnuCOBOL)

1. Clone or unpack GnuCOBOL somewhere, e.g. `C:\repos\gnucobol`.
2. From the LegacyLens repo root, run:

   ```powershell
   python -m backend.ingestion.pipeline run --code-root C:\repos\gnucobol
   ```

3. Use the same Step 5 and Step 6 to start the API and query. No need to change anything else; the vector DB will now search over the full codebase.

---

## If something fails

- **“No module named 'backend'”**  
  Run from the repo root and use `pip install -e .` (Step 3).

- **“Provide a valid --code-root”**  
  Use an absolute path, e.g. `--code-root C:\Users\aaron\Documents\GauntletCode\2-legacylens\sample_cobol`.

- **Connection refused on port 8000**  
  Start the API first (Step 5) and leave it running, then run the query (Step 6).

- **Empty or odd results**  
  Re-run Step 4 so the vector DB is filled, then query again.

---

## After setup: searching the DB and calibrating

- **List what’s in the DB:** open **http://localhost:8000/chunks** in the browser, or run `python scripts/inspect_db.py`.
- **Semantic search:** use POST /query (or `python scripts/query_cli.py "your question"`).
- **Tuning and improving results:** see [CALIBRATION.md](CALIBRATION.md).
