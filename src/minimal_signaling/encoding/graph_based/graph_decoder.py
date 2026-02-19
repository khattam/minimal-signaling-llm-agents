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
        
        # Calculate target output based on graph complexity
        # Aim for ~25-30 tokens per node (balanced between detail and conciseness)
        target_tokens = graph.node_count() * 28
        
        # Build prompt with emphasis on balanced output
        completeness_instruction = f"""
CRITICAL OUTPUT REQUIREMENTS:
- Target output length: approximately {target_tokens} tokens
- Include ALL key information from each node
- Provide necessary context for numbers and metrics
- Use complete, well-formed sentences
- Be direct and factual - no verbose explanations
- Don't repeat information
- Balance detail with conciseness"""
        
        prompt = f"""You are reconstructing a message from a semantic graph. Convert the structured information back into natural language.

The graph contains these semantic elements:
- Intents/Actions: {', '.join(all_intents) if all_intents else intent}
- Entities: {', '.join(entities) if entities else 'None'}
- Attributes: {', '.join(attributes) if attributes else 'None'}
- Details: {', '.join(details) if details else 'None'}
- Constraints: {', '.join(constraints) if constraints else 'None'}
- Outcomes: {', '.join(outcomes) if outcomes else 'None'}

{completeness_instruction}

CRITICAL RULES:
1. Be FACTUAL - only state what's in the graph
2. Keep numbers EXACT - if graph says "23% decline", say exactly that
3. NO fluff phrases like "I would like to", "As we approach"
4. NO hallucinations - don't invent details
5. Provide ADEQUATE DETAIL for each element - don't just list, explain
6. Maintain ORIGINAL tone
7. Write in complete, well-formed paragraphs with proper context

Reconstruct the message using the information provided above. Aim for approximately {target_tokens} tokens.
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
