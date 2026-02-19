"""API Server for Hierarchical Adaptive Encoding Dashboard."""

import asyncio
import json
import os
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# Load environment variables
load_dotenv()

from minimal_signaling.groq_client import GroqClient
from minimal_signaling.semantic_judge import SemanticJudge
from minimal_signaling.msp_decoder import MSPDecoder
from minimal_signaling.encoding.hierarchical_adaptive_encoder import HierarchicalAdaptiveEncoder
from minimal_signaling.tokenization import TiktokenTokenizer


app = FastAPI(title="Hierarchical Adaptive Encoding API")

# CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Data directory
DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)


class EncodeRequest(BaseModel):
    """Request to encode a message."""
    text: str
    target_similarity: float = 0.80
    max_iterations: int = 5


class RunMetadata(BaseModel):
    """Metadata for a saved run."""
    run_id: str
    timestamp: str
    success: bool
    iterations: int
    final_similarity: float
    compression_ratio: float
    original_tokens: int
    final_tokens: int


@app.get("/api/runs")
async def list_runs() -> List[RunMetadata]:
    """List all saved runs."""
    runs = []
    
    for file in DATA_DIR.glob("run_*.json"):
        try:
            with open(file) as f:
                data = json.load(f)
                runs.append(RunMetadata(
                    run_id=file.stem,
                    timestamp=data["metadata"]["timestamp"],
                    success=data["success"],
                    iterations=data["iterations"],
                    final_similarity=data["final_similarity"],
                    compression_ratio=data["compression_ratio"],
                    original_tokens=data["original_tokens"],
                    final_tokens=data["final_tokens"]
                ))
        except Exception as e:
            print(f"Error loading {file}: {e}")
            continue
    
    # Sort by timestamp descending
    runs.sort(key=lambda r: r.timestamp, reverse=True)
    return runs


@app.get("/api/runs/{run_id}")
async def get_run(run_id: str) -> Dict[str, Any]:
    """Get detailed data for a specific run."""
    file_path = DATA_DIR / f"{run_id}.json"
    
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Run not found")
    
    with open(file_path) as f:
        return json.load(f)


@app.post("/api/encode")
async def encode_message(request: EncodeRequest) -> Dict[str, Any]:
    """Encode a message and save the results."""
    try:
        print(f"Received encode request: {len(request.text)} chars")
        # Initialize components
        groq = GroqClient()
        judge = SemanticJudge(threshold=request.target_similarity)
        decoder = MSPDecoder(groq)
        tokenizer = TiktokenTokenizer()
        
        encoder = HierarchicalAdaptiveEncoder(
            groq_client=groq,
            judge=judge,
            decoder=decoder,
            max_iterations=request.max_iterations,
            target_similarity=request.target_similarity
        )
        
        # Run encoding
        result = await encoder.encode_with_refinement(request.text)
        
        # Generate run ID
        run_id = f"run_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
        
        # Calculate ACTUAL compression: original tokens vs decoded tokens
        decoded_tokens = tokenizer.count_tokens(result.final_decoded)
        actual_compression_ratio = decoded_tokens / result.original_tokens
        compression_percentage = (1 - actual_compression_ratio) * 100
        
        # Prepare comprehensive data
        output = {
            "metadata": {
                "run_id": run_id,
                "timestamp": datetime.utcnow().isoformat(),
                "test_name": "API Encoding",
                "model": "llama-3.3-70b-versatile"
            },
            "success": result.converged,
            "iterations": result.iterations,
            "original_tokens": result.original_tokens,
            "decoded_tokens": decoded_tokens,
            "signal_tokens": result.signal_tokens,
            "final_tokens": decoded_tokens,  # For backward compatibility
            "compression_ratio": actual_compression_ratio,
            "compression_percentage": compression_percentage,
            "final_similarity": result.final_similarity,
            "target_similarity": request.target_similarity,
            "sections": {
                "count": len(result.final_signal.sections),
                "breakdown": [
                    {
                        "title": sec.title,
                        "importance": sec.importance,
                        "tokens": tokenizer.count_tokens(sec.content),
                        "content_preview": sec.content[:100] + "..." if len(sec.content) > 100 else sec.content
                    }
                    for sec in result.final_signal.sections
                ]
            },
            "iteration_history": [
                {
                    "iteration": step.iteration,
                    "similarity": step.similarity_score,
                    "tokens": step.signal_tokens,
                    "compression": step.signal_tokens / result.original_tokens,
                    "section_importances": [
                        {
                            "title": sec.title,
                            "importance": sec.importance,
                            "key_concepts": sec.key_concepts
                        }
                        for sec in step.section_importances
                    ] if step.iteration == 1 else None,
                    "feedback": step.feedback
                }
                for step in result.refinement_history
            ],
            "texts": {
                "original": result.original_text,
                "final_decoded": result.final_decoded,
                "final_signal_json": result.final_signal.model_dump_json(indent=2)
            }
        }
        
        # Save to file
        file_path = DATA_DIR / f"{run_id}.json"
        with open(file_path, "w") as f:
            json.dump(output, f, indent=2)
        
        print(f"✅ Saved run to {file_path}")
        
        return output
        
    except Exception as e:
        import traceback
        print(f"Error encoding: {e}")
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/api/runs/{run_id}")
async def delete_run(run_id: str):
    """Delete a saved run."""
    file_path = DATA_DIR / f"{run_id}.json"
    
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Run not found")
    
    file_path.unlink()
    return {"message": "Run deleted"}


# Serve frontend static files in production
if os.path.exists("frontend/dist"):
    app.mount("/", StaticFiles(directory="frontend/dist", html=True), name="static")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
