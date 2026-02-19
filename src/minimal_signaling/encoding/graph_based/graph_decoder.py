"""Graph Decoder - Reconstructs natural language from semantic graphs."""

from ...groq_client import GroqClient
from .semantic_graph import SemanticGraph, NodeType


GRAPH_DECODER_PROMPT = """You are reconstructing a message from a semantic graph. Your job is to convert the structured information back into natural language.

The graph contains these semantic elements:
- Intent: {intent}
- Entities: {entities}
- Attributes: {attributes}
- Details: {details}
- Constraints: {constraints}
- Outcomes: {outcomes}

CRITICAL RULES:
1. Be FACTUAL - only state what's in the graph, nothing more
2. Keep numbers EXACT - if graph says "23% decline", say exactly that
3. NO fluff phrases like "I would like to", "As we approach", "Overall"
4. NO hallucinations - don't invent connections or details
5. Be CONCISE - get straight to the point
6. Maintain ORIGINAL tone - if it was urgent/casual/formal, keep that tone
7. List facts clearly - don't try to make it sound "professional" if it wasn't

Reconstruct the message using ONLY the information provided above.
Output ONLY the reconstructed message."""


class GraphDecoder:
    """Decodes semantic graphs back to natural language."""
    
    def __init__(self, groq_client: GroqClient):
        """Initialize decoder.
        
        Args:
            groq_client: Groq client for LLM calls
        """
        self.client = groq_client
    
    async def decode(self, graph: SemanticGraph, style: str = "professional") -> str:
        """Decode graph to natural language.
        
        Args:
            graph: Semantic graph to decode
            style: Output style (professional, casual, technical)
            
        Returns:
            Natural language text
        """
        # Extract information by node type
        intent = self._get_intent(graph)
        entities = self._get_nodes_content(graph, NodeType.ENTITY)
        attributes = self._get_nodes_content(graph, NodeType.ATTRIBUTE)
        details = self._get_nodes_content(graph, NodeType.DETAIL)
        constraints = self._get_nodes_content(graph, NodeType.CONSTRAINT)
        outcomes = self._get_nodes_content(graph, NodeType.OUTCOME)
        
        # Get all intents (not just root)
        all_intents = [node.content for node in graph.get_nodes_by_type(NodeType.INTENT)]
        
        # Get ALL node contents to force the model to address each one
        all_nodes_text = []
        for node in graph.nodes.values():
            all_nodes_text.append(f"- {node.content} ({node.node_type.value})")
        
        # Build prompt with explicit node list
        prompt = f"""You are reconstructing a message from a semantic graph with {graph.node_count()} nodes.

Here are ALL the nodes you MUST include in your reconstruction:

{chr(10).join(all_nodes_text)}

CRITICAL RULES:
1. You MUST address EVERY node listed above in your reconstruction
2. Keep numbers EXACT - if a node says "23% decline", say exactly that
3. NO fluff phrases like "I would like to", "As we approach"
4. NO hallucinations - only use information from the nodes above
5. Write in complete, well-formed paragraphs that flow naturally
6. Provide context for each piece of information
7. Explain relationships between concepts

Reconstruct the complete message ensuring you include information from ALL {graph.node_count()} nodes listed above.
Output ONLY the reconstructed message."""
        
        # Decode using LLM
        response = await self.client.chat(
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": "Reconstruct the message with appropriate detail."}
            ],
            temperature=0.0
        )
        
        return response.strip()
    
    def _get_intent(self, graph: SemanticGraph) -> str:
        """Get the intent from the graph."""
        if graph.root_id:
            root = graph.get_node(graph.root_id)
            if root:
                return root.content
        
        # Fallback: find any intent node
        intent_nodes = graph.get_nodes_by_type(NodeType.INTENT)
        if intent_nodes:
            return intent_nodes[0].content
        
        return "QUERY"
    
    def _get_nodes_content(self, graph: SemanticGraph, node_type: NodeType) -> list:
        """Get content from all nodes of a specific type."""
        nodes = graph.get_nodes_by_type(node_type)
        return [node.content for node in nodes if node.content]
