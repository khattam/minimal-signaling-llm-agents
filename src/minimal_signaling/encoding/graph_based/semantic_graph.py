"""Semantic Graph representation for message compression.

A semantic graph represents a message as nodes (concepts) and edges (relationships).
Each node has an analytically-calculated importance score based on information theory.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Dict, Any, Set
import uuid


class NodeType(str, Enum):
    """Types of semantic nodes in the graph."""
    INTENT = "intent"           # What action is being requested
    ENTITY = "entity"           # Who/what is involved
    ATTRIBUTE = "attribute"     # Properties, quantities, timeframes
    DETAIL = "detail"           # Context, explanations, specifics
    CONSTRAINT = "constraint"   # Requirements, limitations
    OUTCOME = "outcome"         # Expected results, goals


@dataclass
class SemanticNode:
    """A node in the semantic graph.
    
    Attributes:
        id: Unique identifier
        content: The semantic content
        node_type: Type of semantic information
        importance: Analytically calculated importance score (0-1)
        entropy: Information content in bits
        metadata: Additional structured data
    """
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    content: str = ""
    node_type: NodeType = NodeType.DETAIL
    importance: float = 0.0
    entropy: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __hash__(self):
        return hash(self.id)
    
    def __eq__(self, other):
        if not isinstance(other, SemanticNode):
            return False
        return self.id == other.id


@dataclass
class SemanticEdge:
    """An edge connecting two nodes in the semantic graph.
    
    Attributes:
        source: Source node ID
        target: Target node ID
        relation: Type of relationship (e.g., "has_attribute", "requires", "leads_to")
        weight: Strength of relationship (0-1)
    """
    source: str
    target: str
    relation: str = "related_to"
    weight: float = 1.0


class SemanticGraph:
    """A graph representation of semantic information.
    
    The graph captures the structure and relationships in a message,
    with each node having an importance score for compression.
    """
    
    def __init__(self):
        self.nodes: Dict[str, SemanticNode] = {}
        self.edges: List[SemanticEdge] = []
        self.root_id: str = None
        self.original_text: str = ""
        self.original_tokens: int = 0
    
    def add_node(self, node: SemanticNode) -> str:
        """Add a node to the graph."""
        self.nodes[node.id] = node
        return node.id
    
    def add_edge(self, source_id: str, target_id: str, relation: str = "related_to", weight: float = 1.0):
        """Add an edge between two nodes."""
        if source_id in self.nodes and target_id in self.nodes:
            edge = SemanticEdge(source_id, target_id, relation, weight)
            self.edges.append(edge)
    
    def get_node(self, node_id: str) -> SemanticNode:
        """Get a node by ID."""
        return self.nodes.get(node_id)
    
    def get_neighbors(self, node_id: str) -> List[SemanticNode]:
        """Get all nodes connected to the given node."""
        neighbors = []
        for edge in self.edges:
            if edge.source == node_id:
                neighbors.append(self.nodes[edge.target])
            elif edge.target == node_id:
                neighbors.append(self.nodes[edge.source])
        return neighbors
    
    def get_nodes_by_type(self, node_type: NodeType) -> List[SemanticNode]:
        """Get all nodes of a specific type."""
        return [node for node in self.nodes.values() if node.node_type == node_type]
    
    def total_entropy(self) -> float:
        """Calculate total entropy (information content) of the graph."""
        return sum(node.entropy for node in self.nodes.values())
    
    def total_importance(self) -> float:
        """Calculate total importance score of the graph."""
        return sum(node.importance for node in self.nodes.values())
    
    def node_count(self) -> int:
        """Get total number of nodes."""
        return len(self.nodes)
    
    def edge_count(self) -> int:
        """Get total number of edges."""
        return len(self.edges)
    
    def get_sorted_nodes(self, by: str = "importance", reverse: bool = True) -> List[SemanticNode]:
        """Get nodes sorted by a metric.
        
        Args:
            by: Metric to sort by ("importance", "entropy", "type")
            reverse: Sort in descending order if True
        """
        if by == "importance":
            return sorted(self.nodes.values(), key=lambda n: n.importance, reverse=reverse)
        elif by == "entropy":
            return sorted(self.nodes.values(), key=lambda n: n.entropy, reverse=reverse)
        elif by == "type":
            return sorted(self.nodes.values(), key=lambda n: n.node_type.value)
        else:
            return list(self.nodes.values())
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert graph to dictionary for serialization."""
        return {
            "nodes": [
                {
                    "id": node.id,
                    "content": node.content,
                    "type": node.node_type.value,
                    "importance": node.importance,
                    "entropy": node.entropy,
                    "metadata": node.metadata
                }
                for node in self.nodes.values()
            ],
            "edges": [
                {
                    "source": edge.source,
                    "target": edge.target,
                    "relation": edge.relation,
                    "weight": edge.weight
                }
                for edge in self.edges
            ],
            "root_id": self.root_id,
            "metrics": {
                "total_entropy": self.total_entropy(),
                "total_importance": self.total_importance(),
                "node_count": self.node_count(),
                "edge_count": self.edge_count(),
                "original_tokens": self.original_tokens
            }
        }
    
    def clone(self) -> 'SemanticGraph':
        """Create a deep copy of the graph."""
        new_graph = SemanticGraph()
        new_graph.original_text = self.original_text
        new_graph.original_tokens = self.original_tokens
        new_graph.root_id = self.root_id
        
        # Copy nodes
        for node in self.nodes.values():
            new_node = SemanticNode(
                id=node.id,
                content=node.content,
                node_type=node.node_type,
                importance=node.importance,
                entropy=node.entropy,
                metadata=node.metadata.copy()
            )
            new_graph.add_node(new_node)
        
        # Copy edges
        for edge in self.edges:
            new_graph.add_edge(edge.source, edge.target, edge.relation, edge.weight)
        
        return new_graph
