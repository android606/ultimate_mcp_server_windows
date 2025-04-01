"""MCP server implementation."""
import asyncio
import json
from typing import Any, Awaitable, Callable, Dict

from llm_gateway.mcp.handlers import knowledge_base
from llm_gateway.utils import get_logger

logger = get_logger(__name__)


class MCPServer:
    """Message Control Protocol server."""
    
    def __init__(self, host: str = "0.0.0.0", port: int = 8090):
        """Initialize MCP server.
        
        Args:
            host: Server host
            port: Server port
        """
        self.host = host
        self.port = port
        self.handlers: Dict[str, Callable[[Dict[str, Any]], Awaitable[Dict[str, Any]]]] = {}
        self.server = None
        
        # Register handlers
        self._register_handlers()
        
        logger.info(f"MCP server initialized on {host}:{port}")
    
    def _register_handlers(self) -> None:
        """Register message handlers."""
        # Knowledge base handlers
        self.handlers["knowledge_base.list"] = knowledge_base.handle_list_knowledge_bases
        self.handlers["knowledge_base.rag_query"] = knowledge_base.handle_rag_query
        self.handlers["knowledge_base.feedback"] = knowledge_base.handle_rag_feedback
        self.handlers["knowledge_base.retrieve"] = knowledge_base.handle_retrieve_documents
        self.handlers["knowledge_base.retrieve_hybrid"] = knowledge_base.handle_retrieve_hybrid
        
        logger.info(f"Registered {len(self.handlers)} message handlers")
    
    async def handle_client(self, reader, writer) -> None:
        """Handle client connection.
        
        Args:
            reader: Stream reader
            writer: Stream writer
        """
        addr = writer.get_extra_info("peername")
        logger.info(f"Client connected: {addr}")
        
        try:
            while True:
                # Read message length (4 bytes)
                length_bytes = await reader.read(4)
                if not length_bytes:
                    break
                    
                # Parse message length
                message_length = int.from_bytes(length_bytes, byteorder="big")
                
                # Read message
                message_bytes = await reader.read(message_length)
                if not message_bytes:
                    break
                    
                # Parse message
                message = json.loads(message_bytes.decode("utf-8"))
                
                # Process message
                response = await self.process_message(message)
                
                # Encode response
                response_bytes = json.dumps(response).encode("utf-8")
                
                # Send response length
                writer.write(len(response_bytes).to_bytes(4, byteorder="big"))
                
                # Send response
                writer.write(response_bytes)
                await writer.drain()
        except Exception as e:
            logger.error(f"Error handling client: {str(e)}")
        finally:
            writer.close()
            await writer.wait_closed()
            logger.info(f"Client disconnected: {addr}")
    
    async def process_message(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """Process message.
        
        Args:
            message: Message to process
            
        Returns:
            Response message
        """
        # Extract message type
        message_type = message.get("type")
        if not message_type:
            return {"status": "error", "error": "Missing message type"}
            
        # Get handler
        handler = self.handlers.get(message_type)
        if not handler:
            return {"status": "error", "error": f"Unknown message type: {message_type}"}
            
        try:
            # Handle message
            return await handler(message)
        except Exception as e:
            logger.error(f"Error processing message: {str(e)}")
            return {"status": "error", "error": f"Error processing message: {str(e)}"}
    
    async def start(self) -> None:
        """Start MCP server."""
        self.server = await asyncio.start_server(
            self.handle_client, self.host, self.port
        )
        
        logger.info(f"MCP server started on {self.host}:{self.port}")
        
        async with self.server:
            await self.server.serve_forever()
    
    async def stop(self) -> None:
        """Stop MCP server."""
        if self.server:
            self.server.close()
            await self.server.wait_closed()
            logger.info("MCP server stopped")


async def run_server() -> None:
    """Run MCP server."""
    server = MCPServer()
    try:
        await server.start()
    except KeyboardInterrupt:
        logger.info("Server interrupted")
    finally:
        await server.stop()


def main() -> None:
    """Main entry point."""
    asyncio.run(run_server())


if __name__ == "__main__":
    main() 