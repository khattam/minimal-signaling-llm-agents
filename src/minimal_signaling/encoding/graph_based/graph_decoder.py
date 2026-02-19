"""Graph Decoder - Reconstructs natural language from semantic graphs."""

from ...groq_client import GroqClient
from .semantic_graph import SemanticGraph, NodeType


GRAPH_DECODER_PROMPT = """You are a semantic decoder. Reconstruct the original message from the semantic graph.

The graph contains:
- Intent: {intent}
- Entities: {entities}
- Attributes: {attributes}
- Details: {details}
- Constraints: {constraints}
- Outcomes: {outcomes}

CRITICAL RULES:
1. Be CONCISE and FACTUAL - no fluff, no introductions, no conclusions
2. Use EXACT numbers and dates from the graph - do not invent or mix up values
3. Preserve the ORIGINAL TONE and STRUCTURE - don't make it sound formal if it wasn't
4. Include ALL information from the graph, but ONLY what's in the graph
5. Do NOT add phrases like "As we approach", "I would like to", "Overall", etc.
6. Do NOT hallucinate or infer information not present in the graph

Output ONLY the reconstructed message, no JSON or explanation."""


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
        
        # Build prompt
        prompt = GRAPH_DECODER_PROMPT.format(
            intent=intent,
            entities=", ".join(entities) if entities else "None",
            attributes=", ".join(attributes) if attributes else "None",
            details=", ".join(details) if details else "None",
            constraints=", ".join(constraints) if constraints else "None",
            outcomes=", ".join(outcomes) if outcomes else "None"
        )
        
        # Add style guidance
        style_guidance = {
            "professional": "Use formal, business-appropriate language.",
            "casual": "Use friendly, conversational language.",
            "technical": "Use precise, technical language with details."
        }
        prompt += f"\n\nStyle: {style_guidance.get(style, style_guidance['professional'])}"
        
        # Decode using LLM
        response = await self.client.chat(
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": "Reconstruct the message."}
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
