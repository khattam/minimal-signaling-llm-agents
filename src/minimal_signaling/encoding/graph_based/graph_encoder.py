"""Graph Encoder - Converts natural language to semantic graph using SOTA NLP tools.

Uses:
- spaCy for entity extraction and dependency parsing
- LLM for semantic understanding
- NetworkX for graph operations
- Information theory for importance scoring
"""

import json
import math
import uuid
from collections import Counter
from typing import List, Dict, Any
import spacy

from ...groq_client import GroqClient
from ...tokenization import TiktokenTokenizer
from .semantic_graph import SemanticGraph, SemanticNode, NodeType


GRAPH_EXTRACTION_PROMPT = """You are a semantic graph extractor. Analyze the message and extract semantic nodes with their relationships.

You MUST output ONLY valid JSON in this exact structure (no other text):
{
  "nodes": [
    {
      "id": "unique_id",
      "content": "the actual content",
      "type": "intent/entity/attribute/detail/constraint/outcome",
      "importance": "critical/high/medium/low"
    }
  ],
  "edges": [
    {
      "source": "node_id",
      "target": "node_id", 
      "relation": "requires/causes/relates_to/has_attribute/constrains/leads_to"
    }
  ]
}

CRITICAL RULES:
- Output MUST be valid JSON only
- No markdown, no code blocks, no explanations
- Start with { and end with }
- Create multiple intent nodes if there are multiple actions requested
- Connect related nodes to EACH OTHER, not just to intent
- Example: "23% decline" (attribute) should connect to "enterprise segment" (entity)
- Example: "$500K budget" (constraint) should connect to "remediation plan" (outcome)
- Be thorough and capture ALL relationships between concepts"""


class GraphEncoder:
    """Encodes natural language into semantic graphs using SOTA NLP tools."""
    
    def __init__(self, groq_client: GroqClient, use_spacy: bool = True):
        """Initialize encoder.
        
        Args:
            groq_client: Groq client for LLM calls
            use_spacy: Whether to use spaCy for entity extraction (requires model download)
        """
        self.client = groq_client
        self.tokenizer = TiktokenTokenizer()
        self.use_spacy = use_spacy
        
        # Try to load spaCy model
        self.nlp = None
        if use_spacy:
            try:
                self.nlp = spacy.load("en_core_web_sm")
            except OSError:
                print("spaCy model not found. Run: python -m spacy download en_core_web_sm")
                self.use_spacy = False
    
    async def encode(self, text: str) -> SemanticGraph:
        """Encode text into a semantic graph.
        
        Args:
            text: Natural language input
            
        Returns:
            SemanticGraph with nodes and edges
        """
        graph = SemanticGraph()
        graph.original_text = text
        graph.original_tokens = self.tokenizer.count_tokens(text)
        
        # Extract semantic structure using LLM
        structure = await self._extract_structure(text)
        
        # Build graph from structure
        self._build_graph(graph, structure, text)
        
        # DISABLED: SpaCy enhancement adds too many isolated numbers/acronyms
        # if self.use_spacy and self.nlp:
        #     self._enhance_with_spacy(graph, text)
        
        # Calculate importance scores
        self._calculate_importance(graph, text)
        
        return graph
    
    async def _extract_structure(self, text: str) -> Dict[str, Any]:
        """Use LLM to extract semantic structure."""
        response = await self.client.chat(
            messages=[
                {"role": "system", "content": GRAPH_EXTRACTION_PROMPT},
                {"role": "user", "content": f"Extract semantic graph from this message:\n\n{text}"}
            ],
            json_mode=True,
            temperature=0.0
        )
        
        try:
            # Try to parse the response
            # Some models might wrap JSON in markdown code blocks
            response = response.strip()
            if response.startswith("```"):
                # Remove markdown code blocks
                lines = response.split("\n")
                response = "\n".join(lines[1:-1]) if len(lines) > 2 else response
                response = response.replace("```json", "").replace("```", "").strip()
            
            return json.loads(response)
        except json.JSONDecodeError as e:
            print(f"Warning: JSON decode failed: {e}")
            print(f"Response was: {response[:200]}...")
            # Fallback: create minimal graph
            return {
                "nodes": [
                    {"id": "intent_1", "content": "QUERY", "type": "intent", "importance": "high"}
                ],
                "edges": []
            }
    
    def _build_graph(self, graph: SemanticGraph, structure: Dict[str, Any], original_text: str):
        """Build graph from extracted structure."""
        # Map to track node IDs
        node_map = {}
        
        # Valid node types
        valid_types = {t.value for t in NodeType}
        
        # Add all nodes
        for node_data in structure.get("nodes", []):
            node_type_str = node_data.get("type", "detail")
            
            # Map common variations to valid types
            if node_type_str not in valid_types:
                type_mapping = {
                    "action": "intent",
                    "event": "detail",
                    "requirement": "constraint",
                    "goal": "outcome",
                    "metric": "attribute",
                    "person": "entity",
                    "organization": "entity",
                    "system": "entity"
                }
                node_type_str = type_mapping.get(node_type_str, "detail")
            
            node = SemanticNode(
                id=node_data.get("id", str(uuid.uuid4())),
                content=node_data.get("content", ""),
                node_type=NodeType(node_type_str),
                entropy=self._calculate_entropy(node_data.get("content", "")),
                metadata={"llm_importance": node_data.get("importance", "medium")}
            )
            node_id = graph.add_node(node)
            node_map[node_data.get("id")] = node_id
            
            # Set root to first intent node
            if node.node_type == NodeType.INTENT and not graph.root_id:
                graph.root_id = node_id
        
        # Add all edges
        for edge_data in structure.get("edges", []):
            source_id = node_map.get(edge_data.get("source"))
            target_id = node_map.get(edge_data.get("target"))
            if source_id and target_id:
                graph.add_edge(
                    source_id, 
                    target_id, 
                    edge_data.get("relation", "related_to")
                )
    
    def _enhance_with_spacy(self, graph: SemanticGraph, text: str):
        """Enhance graph with spaCy NLP analysis."""
        doc = self.nlp(text)
        
        # Extract named entities that might have been missed
        for ent in doc.ents:
            # Check if this entity is already in the graph
            existing = any(
                ent.text.lower() in node.content.lower() 
                for node in graph.nodes.values()
            )
            if not existing:
                node = SemanticNode(
                    content=ent.text,
                    node_type=NodeType.ENTITY,
                    entropy=self._calculate_entropy(ent.text),
                    metadata={"spacy_label": ent.label_}
                )
                node_id = graph.add_node(node)
                if graph.root_id:
                    graph.add_edge(graph.root_id, node_id, "has_entity")
    
    def _calculate_entropy(self, text: str) -> float:
        """Calculate Shannon entropy of text."""
        if not text:
            return 0.0
        
        char_counts = Counter(text.lower())
        total_chars = len(text)
        
        if total_chars == 0:
            return 0.0
        
        entropy = 0.0
        for count in char_counts.values():
            p = count / total_chars
            if p > 0:
                entropy -= p * math.log2(p)
        
        # Scale by length
        return entropy * len(text) / 8  # Approximate bytes
    
    def _calculate_importance(self, graph: SemanticGraph, original_text: str):
        """Calculate importance scores for all nodes using information theory."""
        original_words = set(original_text.lower().split())
        
        # Type-based base importance
        type_weights = {
            NodeType.INTENT: 1.0,
            NodeType.OUTCOME: 0.9,
            NodeType.CONSTRAINT: 0.85,
            NodeType.ENTITY: 0.75,
            NodeType.ATTRIBUTE: 0.65,
            NodeType.DETAIL: 0.5
        }
        
        for node in graph.nodes.values():
            # Base importance from type
            base_importance = type_weights.get(node.node_type, 0.5)
            
            # Specificity: how unique is this content?
            node_words = set(node.content.lower().split())
            overlap = len(node_words & original_words)
            specificity = overlap / len(node_words) if node_words else 0
            
            # Combine factors
            node.importance = base_importance * (0.5 + 0.5 * specificity)
            
            # Boost for numbers (highly specific)
            if any(char.isdigit() for char in node.content):
                node.importance *= 1.2
            
            # Cap at 1.0
            node.importance = min(node.importance, 1.0)
