"""WebSocket server for real-time pipeline updates."""

import asyncio
import json
from typing import Set
from datetime import datetime

from websockets.server import serve, WebSocketServerProtocol
from websockets.exceptions import ConnectionClosed

from .events import EventPayload, PipelineEvent, AsyncEventEmitter


class WebSocketServer:
    """WebSocket server for broadcasting pipeline events to clients.
    
    Clients connect to receive real-time updates about pipeline execution,
    including compression progress, extraction results, and metrics.
    """
    
    def __init__(self, host: str = "localhost", port: int = 8081):
        """Initialize the WebSocket server.
        
        Args:
            host: Host to bind to
            port: Port to listen on
        """
        self.host = host
        self.port = port
        self._clients: Set[WebSocketServerProtocol] = set()
        self._server = None
        self._running = False
    
    async def _handler(self, websocket: WebSocketServerProtocol) -> None:
        """Handle a WebSocket connection.
        
        Args:
            websocket: WebSocket connection
        """
        # Register client
        self._clients.add(websocket)
        
        try:
            # Send welcome message
            await websocket.send(json.dumps({
                "type": "connected",
                "timestamp": datetime.utcnow().isoformat(),
                "message": "Connected to minimal-signaling pipeline"
            }))
            
            # Keep connection alive and handle incoming messages
            async for message in websocket:
                # For now, we just acknowledge messages
                # In the future, this could handle commands
                try:
                    data = json.loads(message)
                    if data.get("type") == "ping":
                        await websocket.send(json.dumps({
                            "type": "pong",
                            "timestamp": datetime.utcnow().isoformat()
                        }))
                except json.JSONDecodeError:
                    pass
                    
        except ConnectionClosed:
            pass
        finally:
            # Unregister client
            self._clients.discard(websocket)
    
    async def broadcast(self, payload: EventPayload) -> None:
        """Broadcast an event to all connected clients.
        
        Args:
            payload: Event payload to broadcast
        """
        if not self._clients:
            return
        
        # Serialize payload
        message = json.dumps({
            "type": "event",
            "event": payload.event.value,
            "timestamp": payload.timestamp.isoformat(),
            "data": payload.data
        })
        
        # Broadcast to all clients
        disconnected = set()
        for client in self._clients:
            try:
                await client.send(message)
            except ConnectionClosed:
                disconnected.add(client)
        
        # Remove disconnected clients
        self._clients -= disconnected
    
    def broadcast_sync(self, payload: EventPayload) -> None:
        """Broadcast synchronously (for non-async contexts).
        
        Args:
            payload: Event payload to broadcast
        """
        try:
            loop = asyncio.get_running_loop()
            asyncio.create_task(self.broadcast(payload))
        except RuntimeError:
            # No running loop
            asyncio.run(self.broadcast(payload))
    
    async def start(self) -> None:
        """Start the WebSocket server."""
        self._server = await serve(
            self._handler,
            self.host,
            self.port
        )
        self._running = True
    
    async def stop(self) -> None:
        """Stop the WebSocket server."""
        if self._server:
            self._server.close()
            await self._server.wait_closed()
            self._running = False
    
    @property
    def is_running(self) -> bool:
        """Check if server is running."""
        return self._running
    
    @property
    def client_count(self) -> int:
        """Get number of connected clients."""
        return len(self._clients)


class WebSocketEventBridge:
    """Bridge between SyncEventEmitter and WebSocketServer.
    
    Subscribes to pipeline events and broadcasts them via WebSocket.
    """
    
    def __init__(
        self,
        ws_server: WebSocketServer,
        event_emitter: AsyncEventEmitter
    ):
        """Initialize the bridge.
        
        Args:
            ws_server: WebSocket server to broadcast to
            event_emitter: Event emitter to subscribe to
        """
        self.ws_server = ws_server
        self.event_emitter = event_emitter
        
        # Subscribe to all events
        self.event_emitter.on_all(self._handle_event)
    
    async def _handle_event(self, payload: EventPayload) -> None:
        """Handle an event by broadcasting it.
        
        Args:
            payload: Event payload
        """
        await self.ws_server.broadcast(payload)
    
    def disconnect(self) -> None:
        """Disconnect from event emitter."""
        self.event_emitter.off_all(self._handle_event)
