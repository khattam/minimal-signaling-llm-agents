"""FastAPI-based dashboard server for the minimal-signaling pipeline."""

import asyncio
from pathlib import Path
from typing import Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel

from .config import MediatorConfig
from .mediator import Mediator
from .compression import DistilBARTCompressor
from .extraction import PlaceholderExtractor
from .tokenization import TiktokenTokenizer
from .judge import PlaceholderJudge
from .events import SyncEventEmitter, EventPayload
from .trace import TraceLogger


class ProcessRequest(BaseModel):
    """Request to process a message through the pipeline."""
    message: str
    

class ProcessResponse(BaseModel):
    """Response from processing a message."""
    success: bool
    original_tokens: int
    final_tokens: int
    compression_ratio: float
    passes: int
    keys_extracted: int
    judge_passed: Optional[bool] = None
    duration_ms: float


class DashboardServer:
    """Dashboard server for visualizing pipeline execution.
    
    Provides:
    - REST API for sending messages to the pipeline
    - WebSocket endpoint for real-time updates
    - Static file serving for the frontend
    """
    
    def __init__(
        self,
        config: MediatorConfig,
        static_dir: Optional[str] = None,
        use_real_compressor: bool = False
    ):
        """Initialize the dashboard server.
        
        Args:
            config: Mediator configuration
            static_dir: Directory for static frontend files
            use_real_compressor: Whether to use DistilBART (slow) or mock
        """
        self.config = config
        self.static_dir = Path(static_dir) if static_dir else None
        
        # Event emitter for real-time updates
        self.event_emitter = SyncEventEmitter()
        
        # Connected WebSocket clients
        self._ws_clients: set[WebSocket] = set()
        
        # Set up event forwarding to WebSocket clients
        self.event_emitter.on_all(self._forward_event)
        
        # Create pipeline components
        self.tokenizer = TiktokenTokenizer()
        
        if use_real_compressor:
            from .compression import DistilBARTCompressor
            self.compressor = DistilBARTCompressor()
        else:
            # Use mock compressor for faster testing
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
        
        # Create mediator
        self.mediator = Mediator(
            config=config,
            compressor=self.compressor,
            extractor=self.extractor,
            tokenizer=self.tokenizer,
            judge=self.judge,
            event_emitter=self.event_emitter
        )
        
        # Trace logger
        self.trace_logger = TraceLogger()
        
        # Create FastAPI app
        self.app = self._create_app()
    
    def _forward_event(self, payload: EventPayload) -> None:
        """Forward event to all WebSocket clients."""
        import json
        
        message = json.dumps({
            "type": "event",
            "event": payload.event.value,
            "timestamp": payload.timestamp.isoformat(),
            "data": payload.data
        })
        
        # Schedule broadcast (we're in sync context)
        disconnected = set()
        for ws in self._ws_clients:
            try:
                # Use asyncio to send
                asyncio.create_task(ws.send_text(message))
            except Exception:
                disconnected.add(ws)
        
        self._ws_clients -= disconnected
    
    def _create_app(self) -> FastAPI:
        """Create the FastAPI application."""
        
        @asynccontextmanager
        async def lifespan(app: FastAPI):
            # Startup
            yield
            # Shutdown
            pass
        
        app = FastAPI(
            title="Minimal Signaling Dashboard",
            description="Real-time visualization of the minimal-signaling pipeline",
            version="0.1.0",
            lifespan=lifespan
        )
        
        @app.get("/", response_class=HTMLResponse)
        async def root():
            """Serve the dashboard homepage."""
            return """
            <!DOCTYPE html>
            <html>
            <head>
                <title>Minimal Signaling Dashboard</title>
                <style>
                    body { font-family: system-ui; max-width: 800px; margin: 0 auto; padding: 20px; }
                    .status { padding: 10px; border-radius: 5px; margin: 10px 0; }
                    .success { background: #d4edda; }
                    .error { background: #f8d7da; }
                    pre { background: #f5f5f5; padding: 10px; overflow-x: auto; }
                    textarea { width: 100%; height: 100px; }
                    button { padding: 10px 20px; cursor: pointer; }
                    #events { max-height: 300px; overflow-y: auto; }
                </style>
            </head>
            <body>
                <h1>Minimal Signaling Pipeline</h1>
                <p>Send a message through the compression → extraction → judge pipeline.</p>
                
                <h2>Send Message</h2>
                <textarea id="message" placeholder="Enter a message to process..."></textarea>
                <br><button onclick="sendMessage()">Process Message</button>
                
                <h2>Result</h2>
                <div id="result"></div>
                
                <h2>Real-time Events</h2>
                <div id="events"></div>
                
                <script>
                    const ws = new WebSocket('ws://' + window.location.host + '/ws');
                    const eventsDiv = document.getElementById('events');
                    
                    ws.onmessage = (event) => {
                        const data = JSON.parse(event.data);
                        const div = document.createElement('div');
                        div.className = 'status';
                        div.textContent = `[${data.event || data.type}] ${JSON.stringify(data.data || {})}`;
                        eventsDiv.insertBefore(div, eventsDiv.firstChild);
                    };
                    
                    async function sendMessage() {
                        const message = document.getElementById('message').value;
                        const resultDiv = document.getElementById('result');
                        
                        try {
                            const response = await fetch('/api/process', {
                                method: 'POST',
                                headers: {'Content-Type': 'application/json'},
                                body: JSON.stringify({message})
                            });
                            const data = await response.json();
                            
                            resultDiv.innerHTML = `
                                <div class="status ${data.success ? 'success' : 'error'}">
                                    <strong>Success:</strong> ${data.success}<br>
                                    <strong>Original tokens:</strong> ${data.original_tokens}<br>
                                    <strong>Final tokens:</strong> ${data.final_tokens}<br>
                                    <strong>Compression ratio:</strong> ${(data.compression_ratio * 100).toFixed(1)}%<br>
                                    <strong>Passes:</strong> ${data.passes}<br>
                                    <strong>Keys extracted:</strong> ${data.keys_extracted}<br>
                                    <strong>Duration:</strong> ${data.duration_ms.toFixed(2)}ms
                                </div>
                            `;
                        } catch (e) {
                            resultDiv.innerHTML = `<div class="status error">Error: ${e.message}</div>`;
                        }
                    }
                </script>
            </body>
            </html>
            """
        
        @app.post("/api/process", response_model=ProcessResponse)
        async def process_message(request: ProcessRequest):
            """Process a message through the pipeline."""
            result = self.mediator.process(request.message)
            
            original_tokens = self.tokenizer.count_tokens(request.message)
            
            # Log trace
            self.trace_logger.log_trace_from_result(
                original_text=request.message,
                original_tokens=original_tokens,
                result=result,
                config=self.config
            )
            
            return ProcessResponse(
                success=result.success,
                original_tokens=original_tokens,
                final_tokens=result.compression.final_tokens if result.compression else original_tokens,
                compression_ratio=result.compression.total_ratio if result.compression else 1.0,
                passes=result.compression.passes if result.compression else 0,
                keys_extracted=len(result.extraction.keys) if result.extraction else 0,
                judge_passed=result.judge.passed if result.judge else None,
                duration_ms=result.duration_ms
            )
        
        @app.get("/api/config")
        async def get_config():
            """Get current configuration."""
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
                "judge": {
                    "enabled": self.config.judge.enabled
                }
            }
        
        @app.websocket("/ws")
        async def websocket_endpoint(websocket: WebSocket):
            """WebSocket endpoint for real-time updates."""
            await websocket.accept()
            self._ws_clients.add(websocket)
            
            try:
                # Send welcome message
                await websocket.send_json({
                    "type": "connected",
                    "message": "Connected to minimal-signaling pipeline"
                })
                
                # Keep connection alive
                while True:
                    try:
                        data = await websocket.receive_text()
                        # Handle ping/pong
                        if data == '{"type":"ping"}':
                            await websocket.send_json({"type": "pong"})
                    except WebSocketDisconnect:
                        break
            finally:
                self._ws_clients.discard(websocket)
        
        # Mount static files if directory exists
        if self.static_dir and self.static_dir.exists():
            app.mount("/static", StaticFiles(directory=str(self.static_dir)), name="static")
        
        return app
    
    def run(self, host: str = "localhost", port: int = 8080):
        """Run the dashboard server.
        
        Args:
            host: Host to bind to
            port: Port to listen on
        """
        import uvicorn
        uvicorn.run(self.app, host=host, port=port)


def create_dashboard_server(
    config_path: Optional[str] = None,
    use_real_compressor: bool = False
) -> DashboardServer:
    """Create a dashboard server with default or custom config.
    
    Args:
        config_path: Optional path to config file
        use_real_compressor: Whether to use DistilBART
        
    Returns:
        DashboardServer instance
    """
    if config_path:
        config = MediatorConfig.from_yaml(config_path)
    else:
        from .config import CompressionConfig, SemanticKeysConfig, JudgeConfig
        config = MediatorConfig(
            compression=CompressionConfig(
                enabled=True,
                token_budget=50,
                max_recursion=5
            ),
            semantic_keys=SemanticKeysConfig(enabled=True),
            judge=JudgeConfig(enabled=True)
        )
    
    return DashboardServer(config=config, use_real_compressor=use_real_compressor)
