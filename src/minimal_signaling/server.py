"""FastAPI-based dashboard server for the minimal-signaling pipeline."""

import asyncio
import os
from pathlib import Path

# Load .env file from project root
from dotenv import load_dotenv
env_path = Path(__file__).parent.parent.parent / ".env"
load_dotenv(env_path)
from typing import Optional, Any
from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from .config import MediatorConfig
from .mediator import Mediator
from .extraction import PlaceholderExtractor
from .tokenization import TiktokenTokenizer
from .judge import PlaceholderJudge
from .events import SyncEventEmitter, EventPayload
from .trace import TraceLogger


class ProcessRequest(BaseModel):
    """Request to process a message through the pipeline."""
    message: str
    budget: Optional[int] = None


class KeyResponse(BaseModel):
    """A semantic key in the response."""
    type: str
    value: str


class ProcessResponse(BaseModel):
    """Response from processing a message."""
    success: bool
    original_tokens: int
    final_tokens: int
    compression_ratio: float
    passes: int
    keys_extracted: int
    keys: list[KeyResponse] = []
    compressed_text: Optional[str] = None
    judge_passed: Optional[bool] = None
    judge_confidence: Optional[float] = None
    duration_ms: float


# MSP-specific request/response models
class MSPProcessRequest(BaseModel):
    """Request to process a message through MSP pipeline."""
    message: str
    style: str = "professional"


class MSPSignalResponse(BaseModel):
    """MSP signal in response format."""
    version: str
    intent: str
    target: str
    params: dict[str, Any]
    constraints: list[str]
    state: dict[str, Any]
    priority: str
    trace_id: str
    timestamp: str


class MSPProcessResponse(BaseModel):
    """Response from MSP pipeline processing."""
    success: bool
    original_text: str
    signal: MSPSignalResponse
    decoded_text: str
    judge_passed: bool
    judge_confidence: float
    similarity_score: float
    original_tokens: int
    signal_tokens: int
    decoded_tokens: int
    compression_ratio: float
    latency_ms: float
    trace_id: str


class AgentFlowRequest(BaseModel):
    """Request for Agent A → MSP → Agent B flow."""
    agent_a_message: str


class AgentFlowResponse(BaseModel):
    """Response showing full agent communication flow."""
    success: bool
    agent_a_message: str
    agent_a_tokens: int
    signal: MSPSignalResponse
    signal_json: str
    signal_tokens: int
    agent_b_response: str
    agent_b_tokens: int
    compression_ratio: float
    tokens_saved: int
    latency_ms: float


class RefinementStepResponse(BaseModel):
    """One iteration of refinement."""
    iteration: int
    signal_tokens: int
    similarity: float
    feedback: Optional[str]
    intent: str
    target: str


class IterativeFlowRequest(BaseModel):
    """Request for iterative encoding flow."""
    agent_a_message: str
    target_similarity: float = 0.85
    max_iterations: int = 3


class IterativeFlowResponse(BaseModel):
    """Response with full iterative refinement history."""
    success: bool
    agent_a_message: str
    agent_a_tokens: int
    iterations: int
    converged: bool
    refinement_history: list[RefinementStepResponse]
    final_signal: MSPSignalResponse
    final_signal_json: str
    final_signal_tokens: int
    final_similarity: float
    agent_b_response: str
    agent_b_tokens: int
    compression_ratio: float
    tokens_saved: int
    latency_ms: float


# Hierarchical encoding models
class HierarchicalNodeResponse(BaseModel):
    """A node in the hierarchical semantic tree."""
    content: str
    level: str
    node_type: str
    importance: float
    entropy: float
    children: list["HierarchicalNodeResponse"] = []


class ParetoPointResponse(BaseModel):
    """A point on the Pareto frontier."""
    target_similarity: float
    minimum_bits: float
    compression_ratio: float


class HierarchicalEncodeRequest(BaseModel):
    """Request for hierarchical encoding."""
    message: str
    compress_to_k: Optional[int] = None  # Optional: keep top K nodes


class HierarchicalEncodeResponse(BaseModel):
    """Response from hierarchical encoding."""
    success: bool
    original_text: str
    original_tokens: int
    tree: HierarchicalNodeResponse
    total_nodes: int
    total_entropy: float
    total_importance: float
    pareto_frontier: list[ParetoPointResponse]
    theoretical_bound_80: float
    efficiency: float
    compressed_tree: Optional[HierarchicalNodeResponse] = None
    compressed_nodes: Optional[int] = None
    compressed_entropy: Optional[float] = None
    importance_preserved: Optional[float] = None
    latency_ms: float


class DashboardServer:
    """Dashboard server for visualizing pipeline execution."""
    
    def __init__(
        self,
        config: MediatorConfig,
        static_dir: Optional[str] = None,
        use_real_compressor: bool = False
    ):
        self.config = config
        self.static_dir = Path(static_dir) if static_dir else Path(__file__).parent.parent.parent / "frontend" / "dist"
        
        self.event_emitter = SyncEventEmitter()
        self._ws_clients: set[WebSocket] = set()
        self.event_emitter.on_all(self._forward_event)
        
        self.tokenizer = TiktokenTokenizer()
        
        if use_real_compressor:
            from .compression import DistilBARTCompressor
            self.compressor = DistilBARTCompressor()
        else:
            from .interfaces import Compressor
            class MockCompressor(Compressor):
                def compress(self, text: str) -> str:
                    if not text.strip():
                        return text
                    words = text.split()
                    return " ".join(words[: max(1, len(words) // 2)])
            self.compressor = MockCompressor()
        
        self.extractor = PlaceholderExtractor()
        self.judge = PlaceholderJudge() if config.judge.enabled else None
        
        self.mediator = Mediator(
            config=config,
            compressor=self.compressor,
            extractor=self.extractor,
            tokenizer=self.tokenizer,
            judge=self.judge,
            event_emitter=self.event_emitter
        )
        
        self.trace_logger = TraceLogger()
        self.app = self._create_app()
    
    def _forward_event(self, payload: EventPayload) -> None:
        import json
        message = json.dumps({
            "type": "event",
            "event": payload.event.value,
            "timestamp": payload.timestamp.isoformat(),
            "data": payload.data
        })
        disconnected = set()
        for ws in self._ws_clients:
            try:
                asyncio.create_task(ws.send_text(message))
            except Exception:
                disconnected.add(ws)
        self._ws_clients -= disconnected
    
    def _create_app(self) -> FastAPI:
        @asynccontextmanager
        async def lifespan(app: FastAPI):
            yield
        
        app = FastAPI(
            title="Minimal Signaling Dashboard",
            version="0.1.0",
            lifespan=lifespan
        )
        
        # CORS for development
        app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
        @app.post("/api/process", response_model=ProcessResponse)
        async def process_message(request: ProcessRequest):
            if request.budget is not None:
                self.mediator.compression_engine.budget = request.budget
            
            result = self.mediator.process(request.message)
            original_tokens = self.tokenizer.count_tokens(request.message)
            
            self.trace_logger.log_trace_from_result(
                original_text=request.message,
                original_tokens=original_tokens,
                result=result,
                config=self.config
            )
            
            keys = []
            if result.extraction:
                keys = [KeyResponse(type=k.type.value, value=k.value) for k in result.extraction.keys]
            
            return ProcessResponse(
                success=result.success,
                original_tokens=original_tokens,
                final_tokens=result.compression.final_tokens if result.compression else original_tokens,
                compression_ratio=result.compression.total_ratio if result.compression else 1.0,
                passes=result.compression.passes if result.compression else 0,
                keys_extracted=len(keys),
                keys=keys,
                compressed_text=result.compression.compressed_text if result.compression else None,
                judge_passed=result.judge.passed if result.judge else None,
                judge_confidence=result.judge.confidence if result.judge else None,
                duration_ms=result.duration_ms
            )
        
        @app.get("/api/config")
        async def get_config():
            return {
                "compression": {
                    "enabled": self.config.compression.enabled,
                    "token_budget": self.config.compression.token_budget,
                    "max_recursion": self.config.compression.max_recursion
                },
                "semantic_keys": {
                    "enabled": self.config.semantic_keys.enabled,
                    "schema_version": self.config.semantic_keys.schema_version
                },
                "judge": {"enabled": self.config.judge.enabled},
                "msp": {
                    "enabled": os.environ.get("GROQ_API_KEY") is not None
                }
            }
        
        @app.post("/api/msp/process", response_model=MSPProcessResponse)
        async def process_msp(request: MSPProcessRequest):
            """Process message through MSP pipeline (Groq-based)."""
            groq_key = os.environ.get("GROQ_API_KEY")
            if not groq_key:
                raise HTTPException(
                    status_code=503,
                    detail="GROQ_API_KEY not set. Get a free key at https://console.groq.com"
                )
            
            try:
                from .msp_pipeline import MSPPipeline
                from .msp_config import MSPConfig
                
                config = MSPConfig.from_env()
                pipeline = MSPPipeline(config=config, event_emitter=self.event_emitter)
                
                result = await pipeline.process(request.message, style=request.style)
                
                return MSPProcessResponse(
                    success=True,
                    original_text=result.original_text,
                    signal=MSPSignalResponse(
                        version=result.signal.version,
                        intent=result.signal.intent,
                        target=result.signal.target,
                        params=result.signal.params,
                        constraints=result.signal.constraints,
                        state=result.signal.state,
                        priority=result.signal.priority,
                        trace_id=result.signal.trace_id,
                        timestamp=result.signal.timestamp.isoformat()
                    ),
                    decoded_text=result.decoded_text,
                    judge_passed=result.judge.passed,
                    judge_confidence=result.judge.confidence,
                    similarity_score=result.judge.similarity_score,
                    original_tokens=result.metrics.original_tokens,
                    signal_tokens=result.metrics.signal_tokens,
                    decoded_tokens=result.metrics.decoded_tokens,
                    compression_ratio=result.metrics.compression_ratio,
                    latency_ms=result.metrics.latency_ms,
                    trace_id=result.trace_id
                )
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
        
        @app.post("/api/msp/agent-flow", response_model=AgentFlowResponse)
        async def agent_flow(request: AgentFlowRequest):
            """Full Agent A → MSP Signal → Agent B flow."""
            import time
            groq_key = os.environ.get("GROQ_API_KEY")
            if not groq_key:
                raise HTTPException(
                    status_code=503,
                    detail="GROQ_API_KEY not set. Get a free key at https://console.groq.com"
                )
            
            try:
                start_time = time.time()
                
                from .groq_client import GroqClient
                from .msp_encoder import MSPEncoder
                
                groq = GroqClient(api_key=groq_key)
                encoder = MSPEncoder(groq)
                
                # Count Agent A tokens
                agent_a_tokens = self.tokenizer.count_tokens(request.agent_a_message)
                
                # Encode to MSP
                signal = await encoder.encode(request.agent_a_message)
                signal_json = signal.model_dump_json(indent=2)
                signal_tokens = self.tokenizer.count_tokens(signal_json)
                
                # Agent B receives raw JSON and responds
                agent_b_response = await groq.chat(
                    messages=[
                        {"role": "system", "content": "You are an AI assistant. Respond to the incoming message."},
                        {"role": "user", "content": signal_json}
                    ],
                    temperature=0.3
                )
                agent_b_tokens = self.tokenizer.count_tokens(agent_b_response)
                
                latency_ms = (time.time() - start_time) * 1000
                
                return AgentFlowResponse(
                    success=True,
                    agent_a_message=request.agent_a_message,
                    agent_a_tokens=agent_a_tokens,
                    signal=MSPSignalResponse(
                        version=signal.version,
                        intent=signal.intent,
                        target=signal.target,
                        params=signal.params,
                        constraints=signal.constraints,
                        state=signal.state,
                        priority=signal.priority,
                        trace_id=signal.trace_id,
                        timestamp=signal.timestamp.isoformat()
                    ),
                    signal_json=signal_json,
                    signal_tokens=signal_tokens,
                    agent_b_response=agent_b_response,
                    agent_b_tokens=agent_b_tokens,
                    compression_ratio=signal_tokens / agent_a_tokens if agent_a_tokens > 0 else 1.0,
                    tokens_saved=agent_a_tokens - signal_tokens,
                    latency_ms=latency_ms
                )
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
        
        @app.post("/api/msp/iterative-flow", response_model=IterativeFlowResponse)
        async def iterative_flow(request: IterativeFlowRequest):
            """Iterative encoding with semantic feedback loop."""
            import time
            groq_key = os.environ.get("GROQ_API_KEY")
            if not groq_key:
                raise HTTPException(
                    status_code=503,
                    detail="GROQ_API_KEY not set"
                )
            
            try:
                start_time = time.time()
                
                from .groq_client import GroqClient
                from .iterative_encoder import IterativeEncoder
                from .semantic_judge import SemanticJudge
                from .msp_decoder import MSPDecoder
                
                groq = GroqClient(api_key=groq_key)
                judge = SemanticJudge(threshold=request.target_similarity)
                decoder = MSPDecoder(groq)
                
                encoder = IterativeEncoder(
                    groq_client=groq,
                    judge=judge,
                    decoder=decoder,
                    max_iterations=request.max_iterations,
                    target_similarity=request.target_similarity
                )
                
                agent_a_tokens = self.tokenizer.count_tokens(request.agent_a_message)
                
                # Run iterative encoding
                result = await encoder.encode_with_refinement(request.agent_a_message)
                
                # Build refinement history for response
                history = [
                    RefinementStepResponse(
                        iteration=step.iteration,
                        signal_tokens=step.signal_tokens,
                        similarity=step.similarity_score,
                        feedback=step.feedback,  # Full feedback, not truncated
                        intent=step.signal.intent,
                        target=step.signal.target
                    )
                    for step in result.refinement_history
                ]
                
                signal_json = result.final_signal.model_dump_json(indent=2)
                
                # Agent B receives final signal
                agent_b_response = await groq.chat(
                    messages=[
                        {"role": "system", "content": "You are an AI assistant. Respond to the incoming message."},
                        {"role": "user", "content": signal_json}
                    ],
                    temperature=0.3
                )
                agent_b_tokens = self.tokenizer.count_tokens(agent_b_response)
                
                latency_ms = (time.time() - start_time) * 1000
                
                return IterativeFlowResponse(
                    success=True,
                    agent_a_message=request.agent_a_message,
                    agent_a_tokens=agent_a_tokens,
                    iterations=result.iterations,
                    converged=result.converged,
                    refinement_history=history,
                    final_signal=MSPSignalResponse(
                        version=result.final_signal.version,
                        intent=result.final_signal.intent,
                        target=result.final_signal.target,
                        params=result.final_signal.params,
                        constraints=result.final_signal.constraints,
                        state=result.final_signal.state,
                        priority=result.final_signal.priority,
                        trace_id=result.final_signal.trace_id,
                        timestamp=result.final_signal.timestamp.isoformat()
                    ),
                    final_signal_json=signal_json,
                    final_signal_tokens=result.signal_tokens,
                    final_similarity=result.final_similarity,
                    agent_b_response=agent_b_response,
                    agent_b_tokens=agent_b_tokens,
                    compression_ratio=result.signal_tokens / agent_a_tokens if agent_a_tokens > 0 else 1.0,
                    tokens_saved=agent_a_tokens - result.signal_tokens,
                    latency_ms=latency_ms
                )
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
        
        @app.post("/api/msp/iterative-flow-stream")
        async def iterative_flow_stream(request: IterativeFlowRequest):
            """Iterative encoding with real-time SSE streaming of pipeline stages."""
            import time
            import json as json_module
            
            groq_key = os.environ.get("GROQ_API_KEY")
            if not groq_key:
                raise HTTPException(status_code=503, detail="GROQ_API_KEY not set")
            
            async def event_generator():
                try:
                    start_time = time.time()
                    event_queue: asyncio.Queue = asyncio.Queue()
                    
                    from .groq_client import GroqClient
                    from .iterative_encoder import IterativeEncoder, StageEvent
                    from .semantic_judge import SemanticJudge
                    from .msp_decoder import MSPDecoder
                    
                    def on_stage_change(event: StageEvent):
                        # Put event in queue (sync callback)
                        asyncio.get_event_loop().call_soon_threadsafe(
                            event_queue.put_nowait,
                            event
                        )
                    
                    groq = GroqClient(api_key=groq_key)
                    judge = SemanticJudge(threshold=request.target_similarity)
                    decoder = MSPDecoder(groq)
                    
                    encoder = IterativeEncoder(
                        groq_client=groq,
                        judge=judge,
                        decoder=decoder,
                        max_iterations=request.max_iterations,
                        target_similarity=request.target_similarity,
                        on_stage_change=on_stage_change
                    )
                    
                    agent_a_tokens = self.tokenizer.count_tokens(request.agent_a_message)
                    
                    # Start encoding in background task
                    encoding_task = asyncio.create_task(
                        encoder.encode_with_refinement(request.agent_a_message)
                    )
                    
                    # Stream events as they come
                    while not encoding_task.done():
                        try:
                            event = await asyncio.wait_for(event_queue.get(), timeout=0.1)
                            event_data = {
                                "stage": event.stage.value,
                                "iteration": event.iteration,
                                "similarity": event.similarity,
                                "passed_threshold": event.passed_threshold,
                                "feedback": event.feedback[:100] if event.feedback else None
                            }
                            yield f"data: {json_module.dumps(event_data)}\n\n"
                        except asyncio.TimeoutError:
                            continue
                    
                    # Drain remaining events
                    while not event_queue.empty():
                        event = event_queue.get_nowait()
                        event_data = {
                            "stage": event.stage.value,
                            "iteration": event.iteration,
                            "similarity": event.similarity,
                            "passed_threshold": event.passed_threshold,
                            "feedback": event.feedback[:100] if event.feedback else None
                        }
                        yield f"data: {json_module.dumps(event_data)}\n\n"
                    
                    result = encoding_task.result()
                    
                    # Emit agent_b stage
                    yield f"data: {json_module.dumps({'stage': 'agent_b', 'iteration': result.iterations})}\n\n"
                    
                    signal_json = result.final_signal.model_dump_json(indent=2)
                    
                    # Agent B response
                    agent_b_response = await groq.chat(
                        messages=[
                            {"role": "system", "content": "You are an AI assistant. Respond to the incoming message."},
                            {"role": "user", "content": signal_json}
                        ],
                        temperature=0.3
                    )
                    agent_b_tokens = self.tokenizer.count_tokens(agent_b_response)
                    
                    latency_ms = (time.time() - start_time) * 1000
                    
                    # Build final response
                    history = [
                        {
                            "iteration": step.iteration,
                            "signal_tokens": step.signal_tokens,
                            "similarity": step.similarity_score,
                            "feedback": step.feedback,  # Full feedback
                            "intent": step.signal.intent,
                            "target": step.signal.target
                        }
                        for step in result.refinement_history
                    ]
                    
                    final_response = {
                        "stage": "complete",
                        "result": {
                            "success": True,
                            "agent_a_message": request.agent_a_message,
                            "agent_a_tokens": agent_a_tokens,
                            "iterations": result.iterations,
                            "converged": result.converged,
                            "refinement_history": history,
                            "final_signal": {
                                "version": result.final_signal.version,
                                "intent": result.final_signal.intent,
                                "target": result.final_signal.target,
                                "params": result.final_signal.params,
                                "constraints": result.final_signal.constraints,
                                "state": result.final_signal.state,
                                "priority": result.final_signal.priority,
                                "trace_id": result.final_signal.trace_id,
                                "timestamp": result.final_signal.timestamp.isoformat()
                            },
                            "final_signal_json": signal_json,
                            "final_signal_tokens": result.signal_tokens,
                            "final_similarity": result.final_similarity,
                            "agent_b_response": agent_b_response,
                            "agent_b_tokens": agent_b_tokens,
                            "compression_ratio": result.signal_tokens / agent_a_tokens if agent_a_tokens > 0 else 1.0,
                            "tokens_saved": agent_a_tokens - result.signal_tokens,
                            "latency_ms": latency_ms
                        }
                    }
                    yield f"data: {json_module.dumps(final_response)}\n\n"
                    
                except Exception as e:
                    yield f"data: {json_module.dumps({'stage': 'error', 'error': str(e)})}\n\n"
            
            return StreamingResponse(
                event_generator(),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "X-Accel-Buffering": "no"
                }
            )
        
        @app.post("/api/msp/hierarchical", response_model=HierarchicalEncodeResponse)
        async def hierarchical_encode(request: HierarchicalEncodeRequest):
            """Encode message into hierarchical semantic tree with importance scores."""
            import time
            groq_key = os.environ.get("GROQ_API_KEY")
            if not groq_key:
                raise HTTPException(status_code=503, detail="GROQ_API_KEY not set")
            
            try:
                start_time = time.time()
                
                from .groq_client import GroqClient
                from .hierarchical_encoder import HierarchicalEncoder, HierarchicalCompressor
                
                groq = GroqClient(api_key=groq_key)
                encoder = HierarchicalEncoder(groq)
                
                # Encode to hierarchical signal
                result = await encoder.encode(request.message)
                
                # Convert tree to response format
                def node_to_response(node) -> HierarchicalNodeResponse:
                    return HierarchicalNodeResponse(
                        content=node.content,
                        level=node.level.name,
                        node_type=node.node_type,
                        importance=round(node.importance, 4),
                        entropy=round(node.entropy, 2),
                        children=[node_to_response(c) for c in node.children]
                    )
                
                tree = node_to_response(result.signal.root)
                
                # Get Pareto frontier
                frontier = encoder.bound_calc.pareto_frontier(result.signal)
                pareto = [
                    ParetoPointResponse(
                        target_similarity=p["target_similarity"],
                        minimum_bits=p["minimum_bits"],
                        compression_ratio=p["compression_ratio"]
                    )
                    for p in frontier
                ]
                
                # Optional compression
                compressed_tree = None
                compressed_nodes = None
                compressed_entropy = None
                importance_preserved = None
                
                if request.compress_to_k:
                    compressor = HierarchicalCompressor()
                    compressed = compressor.compress(result.signal, preserve_top_k=request.compress_to_k)
                    compressed_tree = node_to_response(compressed.root)
                    compressed_nodes = compressed.node_count()
                    compressed_entropy = round(compressed.total_entropy(), 2)
                    importance_preserved = round(
                        compressed.total_importance() / result.signal.total_importance(), 4
                    )
                
                latency_ms = (time.time() - start_time) * 1000
                
                return HierarchicalEncodeResponse(
                    success=True,
                    original_text=request.message,
                    original_tokens=result.signal.original_tokens,
                    tree=tree,
                    total_nodes=result.signal.node_count(),
                    total_entropy=round(result.signal.total_entropy(), 2),
                    total_importance=round(result.signal.total_importance(), 4),
                    pareto_frontier=pareto,
                    theoretical_bound_80=round(result.theoretical_bound, 2),
                    efficiency=round(result.efficiency, 4),
                    compressed_tree=compressed_tree,
                    compressed_nodes=compressed_nodes,
                    compressed_entropy=compressed_entropy,
                    importance_preserved=importance_preserved,
                    latency_ms=latency_ms
                )
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
        
        @app.websocket("/ws")
        async def websocket_endpoint(websocket: WebSocket):
            await websocket.accept()
            self._ws_clients.add(websocket)
            try:
                await websocket.send_json({"type": "connected"})
                while True:
                    try:
                        data = await websocket.receive_text()
                        if data == '{"type":"ping"}':
                            await websocket.send_json({"type": "pong"})
                    except WebSocketDisconnect:
                        break
            finally:
                self._ws_clients.discard(websocket)
        
        # Serve frontend static files
        if self.static_dir.exists():
            @app.get("/")
            async def serve_index():
                return FileResponse(self.static_dir / "index.html")
            
            app.mount("/", StaticFiles(directory=str(self.static_dir), html=True), name="static")
        
        return app
    
    def run(self, host: str = "localhost", port: int = 8080):
        import uvicorn
        uvicorn.run(self.app, host=host, port=port)


def create_dashboard_server(
    config_path: Optional[str] = None,
    use_real_compressor: bool = False
) -> DashboardServer:
    if config_path:
        config = MediatorConfig.from_yaml(config_path)
    else:
        from .config import CompressionConfig, SemanticKeysConfig, JudgeConfig
        config = MediatorConfig(
            compression=CompressionConfig(enabled=True, token_budget=50, max_recursion=5),
            semantic_keys=SemanticKeysConfig(enabled=True),
            judge=JudgeConfig(enabled=True)
        )
    return DashboardServer(config=config, use_real_compressor=use_real_compressor)


# Module-level app instance for uvicorn
_server = create_dashboard_server()
app = _server.app
