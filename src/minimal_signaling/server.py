"""FastAPI-based dashboard server for the minimal-signaling pipeline."""

import asyncio
from pathlib import Path
from typing import Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
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
            allow_origins=["http://localhost:5173", "http://localhost:3000"],
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
                "judge": {"enabled": self.config.judge.enabled}
            }
        
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
