"""Graph Encoder - Converts natural language to semantic graph using SOTA NLP tools.

Uses:
- spaCy for entity extraction and dependency parsing
- LLM for semantic understanding
- NetworkX for graph operations
- Information theory for importance scoring
"""

import json
import math
from collections import Counter
from typing import List, Dict, Any
import spacy

from ...groq_client import GroqClient
from ...tokenization import TiktokenTokenizer
from .semantic_graph import SemanticGraph, SemanticNode, NodeType


GRAPH_EXTRACTION_PROMPT = """You are a semantic graph extractor. Analyze the message and extract semantic nodes.

Output JSON with this structure:
{
  "intent": "What action is being requested (ANALYZE/GENERATE/EVALUATE/TRANSFORM/QUERY/RESPOND/DELEGATE/REPORT)",
  "entities": [
    {"content": "...", "type": "person/organization/system/document", "importance": "critical/high/medium/low"}
  ],
  "attributes": [
    {"content": "...", "type": "quantity/timeframe/status/priority", "importance": "critical/high/medium/low"}
  ],
  "details": [
    {"content": "...", "type": "context/explanation/requirement", "importance": "critical/high/medium/low"}
  ],
  "constraints": [
    {"content": "...", "importance": "critical/high/medium/low"}
  ],
  "outcomes": [
    {"content": "expected result or goal", "importance": "critical/high/medium/low"}
  ]
}

Extract ALL important information. Be thorough.
Output ONLY valid JSON."""


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
        
        # Enhance with spaCy if available
        if self.use_spacy and self.nlp:
            self._enhance_with_spacy(graph, text)
        
        # Calculate importance scores
        self._calculate_importance(graph, text)
        
        return graph
    
    async def _extract_structure(self, text: str) -> Dict[str, Any]:
        """Use LLM to extract semantic structure."""
        response = await self.client.chat(
            messages=[
                {"role": "system", "content": GRAPH_EXTRACTION_PROMPT},
                {"role": "user", "content": text}
            ],
            json_mode=True,
            temperature=0.0
        )
        
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            # Fallback
            return {
                "intent": "QUERY",
                "entities": [],
                "attributes": [],
                "details": [],
                "constraints": [],
                "outcomes": []
            }
    
    def _build_graph(self, graph: SemanticGraph, structure: Dict[str, Any], original_text: str):
        """Build graph from extracted structure."""
        # Create intent node (root)
        intent_node = SemanticNode(
            content=structure.get("intent", "QUERY"),
            node_type=NodeType.INTENT,
            entropy=self._calculate_entropy(structure.get("intent", "QUERY"))
        )
        intent_id = graph.add_node(intent_node)
        graph.root_id = intent_id
        
        # Add entities
        for entity in structure.get("entities", []):
            node = SemanticNode(
                content=entity.get("content", ""),
                node_type=NodeType.ENTITY,
                entropy=self._calculate_entropy(entity.get("content", "")),
                metadata={"subtype": entity.get("type", "unknown")}
            )
            node_id = graph.add_node(node)
            graph.add_edge(intent_id, node_id, "has_entity")
        
        # Add attributes
        for attr in structure.get("attributes", []):
            node = SemanticNode(
                content=attr.get("content", ""),
                node_type=NodeType.ATTRIBUTE,
                entropy=self._calculate_entropy(attr.get("content", "")),
                metadata={"subtype": attr.get("type", "unknown")}
            )
            node_id = graph.add_node(node)
            graph.add_edge(intent_id, node_id, "has_attribute")
        
        # Add details
        for detail in structure.get("details", []):
            node = SemanticNode(
                content=detail.get("content", ""),
                node_type=NodeType.DETAIL,
                entropy=self._calculate_entropy(detail.get("content", "")),
                metadata={"subtype": detail.get("type", "unknown")}
            )
            node_id = graph.add_node(node)
            graph.add_edge(intent_id, node_id, "has_detail")
        
        # Add constraints
        for constraint in structure.get("constraints", []):
            node = SemanticNode(
                content=constraint.get("content", ""),
                node_type=NodeType.CONSTRAINT,
                entropy=self._calculate_entropy(constraint.get("content", "")),
            )
            node_id = graph.add_node(node)
            graph.add_edge(intent_id, node_id, "constrained_by")
        
        # Add outcomes
        for outcome in structure.get("outcomes", []):
            node = SemanticNode(
                content=outcome.get("content", ""),
                node_type=NodeType.OUTCOME,
                entropy=self._calculate_entropy(outcome.get("content", "")),
            )
            node_id = graph.add_node(node)
            graph.add_edge(intent_id, node_id, "leads_to")
    
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
