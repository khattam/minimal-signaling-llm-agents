# Hierarchical Adaptive Encoding Dashboard

Full-stack application for running and visualizing hierarchical adaptive encoding experiments.

## Architecture

```
Frontend (React + TypeScript)
    ↓ HTTP API
Backend (FastAPI + Python)
    ↓ Saves to
Data Folder (JSON files)
```

## Features

### Frontend
- **New Run**: Enter text, set parameters, run encoding
- **Live Results**: View metrics, iterations, sections, comparisons
- **Run History**: Browse all past runs from sidebar
- **Visualizations**:
  - Token flow diagram
  - Iteration progress bars
  - Section importance tree
  - Original vs decoded comparison

### Backend
- **POST /api/encode**: Run hierarchical encoding on text
- **GET /api/runs**: List all saved runs
- **GET /api/runs/{run_id}**: Get detailed data for a run
- **DELETE /api/runs/{run_id}**: Delete a run

### Data Persistence
- All runs saved to `data/run_YYYYMMDD_HHMMSS.json`
- Includes full metadata, iteration history, section analysis
- Acts as a database for later analysis

## Running the Dashboard

### 1. Start Backend

```bash
poetry run python -m uvicorn src.minimal_signaling.api_server:app --reload --port 8080
```

Backend runs on: http://localhost:8080

### 2. Start Frontend

```bash
cd frontend
npm run dev
```

Frontend runs on: http://localhost:5173

### 3. Use the Dashboard

1. Click "+ New" in sidebar
2. Enter text to encode
3. Adjust target similarity and max iterations
4. Click "Run Encoding"
5. View results in tabs:
   - **Overview**: Metrics and token flow
   - **Iterations**: Progress and feedback
   - **Sections**: Importance breakdown
   - **Comparison**: Original vs decoded text

## Data Structure

Each run saves:

```json
{
  "metadata": {
    "run_id": "run_20260219_143022",
    "timestamp": "2026-02-19T14:30:22.123Z",
    "model": "llama-3.3-70b-versatile"
  },
  "success": true,
  "iterations": 1,
  "original_tokens": 1763,
  "final_tokens": 727,
  "compression_ratio": 0.412,
  "final_similarity": 0.818,
  "sections": {
    "count": 6,
    "breakdown": [...]
  },
  "iteration_history": [...],
  "texts": {
    "original": "...",
    "final_decoded": "...",
    "final_signal_json": "..."
  }
}
```

## API Examples

### Run Encoding

```bash
curl -X POST http://localhost:8080/api/encode \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Your message here...",
    "target_similarity": 0.80,
    "max_iterations": 5
  }'
```

### List Runs

```bash
curl http://localhost:8080/api/runs
```

### Get Run Details

```bash
curl http://localhost:8080/api/runs/run_20260219_143022
```

### Delete Run

```bash
curl -X DELETE http://localhost:8080/api/runs/run_20260219_143022
```

## Development

### Backend Hot Reload
Backend automatically reloads on code changes (uvicorn --reload)

### Frontend Hot Reload
Frontend automatically reloads on code changes (Vite HMR)

### Data Folder
- Location: `data/`
- Format: `run_YYYYMMDD_HHMMSS.json`
- Gitignored (except example files)

## Tech Stack

- **Backend**: FastAPI, Python 3.11+
- **Frontend**: React 18, TypeScript, Vite
- **Encoding**: Groq (Llama 3.3 70B), sentence-transformers
- **Storage**: JSON files (simple, portable, version-controllable)

## For Your Professor

This dashboard demonstrates:
1. **Real-time encoding**: Run experiments directly from UI
2. **Comprehensive metrics**: Track similarity, compression, iterations
3. **Importance analysis**: Visualize section importance tree
4. **Iterative refinement**: See how feedback improves results
5. **Data persistence**: All runs saved for later analysis

The system achieves 80%+ semantic similarity on long messages (1700+ tokens) with adaptive importance-weighted compression.
