"""Graph Compressor - Compresses semantic graphs by pruning low-importance nodes.

Uses importance scores to make principled compression decisions.
"""

from typing import Optional, Set
import networkx as nx

from .semantic_graph import SemanticGraph, SemanticNode, NodeType


class GraphCompressor:
    """Compresses semantic graphs using importance-based pruning."""
    
    def __init__(self):
        pass
    
    def compress(
        self,
        graph: SemanticGraph,
        target_ratio: float = 0.4,
        preserve_types: Optional[Set[NodeType]] = None
    ) -> SemanticGraph:
        """Compress graph to target ratio by pruning low-importance nodes.
        
        Args:
            graph: Original semantic graph
            target_ratio: Target size as ratio of original (e.g., 0.4 = 40%)
            preserve_types: Node types that should never be pruned
            
        Returns:
            Compressed graph
        """
        if preserve_types is None:
            preserve_types = {NodeType.INTENT}  # Always keep intent
        
        # Calculate target entropy
        target_entropy = graph.total_entropy() * target_ratio
        
        # Get all nodes sorted by importance (descending)
        all_nodes = list(graph.nodes.values())
        
        # Separate must-keep nodes from prunable nodes
        must_keep = [n for n in all_nodes if n.node_type in preserve_types]
        prunable = [n for n in all_nodes if n.node_type not in preserve_types]
        
        # Sort prunable by importance (descending)
        prunable_sorted = sorted(prunable, key=lambda n: n.importance, reverse=True)
        
        # Greedily select nodes until we hit target entropy
        current_entropy = sum(n.entropy for n in must_keep)
        selected_nodes = set(must_keep)
        
        for node in prunable_sorted:
            if current_entropy + node.entropy <= target_entropy:
                selected_nodes.add(node)
                current_entropy += node.entropy
            else:
                # Check if we should include this node anyway (very high importance)
                if node.importance > 0.9 and len(selected_nodes) < 10:
                    selected_nodes.add(node)
                    current_entropy += node.entropy
        
        # Build compressed graph
        compressed = self._build_compressed_graph(graph, selected_nodes)
        
        return compressed
    
    def _build_compressed_graph(
        self,
        original: SemanticGraph,
        selected_nodes: Set[SemanticNode]
    ) -> SemanticGraph:
        """Build a new graph containing only selected nodes."""
        compressed = SemanticGraph()
        compressed.original_text = original.original_text
        compressed.original_tokens = original.original_tokens
        compressed.root_id = original.root_id
        
        # Add selected nodes
        selected_ids = {node.id for node in selected_nodes}
        for node in selected_nodes:
            compressed.add_node(node)
        
        # Add edges between selected nodes
        for edge in original.edges:
            if edge.source in selected_ids and edge.target in selected_ids:
                compressed.add_edge(edge.source, edge.target, edge.relation, edge.weight)
        
        return compressed
    
    def get_compression_stats(self, original: SemanticGraph, compressed: SemanticGraph) -> dict:
        """Get statistics about the compression."""
        return {
            "original_nodes": original.node_count(),
            "compressed_nodes": compressed.node_count(),
            "nodes_removed": original.node_count() - compressed.node_count(),
            "node_retention": compressed.node_count() / original.node_count() if original.node_count() > 0 else 0,
            "original_entropy": original.total_entropy(),
            "compressed_entropy": compressed.total_entropy(),
            "entropy_retention": compressed.total_entropy() / original.total_entropy() if original.total_entropy() > 0 else 0,
            "original_importance": original.total_importance(),
            "compressed_importance": compressed.total_importance(),
            "importance_retention": compressed.total_importance() / original.total_importance() if original.total_importance() > 0 else 0,
        }
    
    def to_networkx(self, graph: SemanticGraph) -> nx.DiGraph:
        """Convert SemanticGraph to NetworkX DiGraph for analysis/visualization.
        
        Returns:
            NetworkX directed graph
        """
        G = nx.DiGraph()
        
        # Add nodes with attributes
        for node in graph.nodes.values():
            G.add_node(
                node.id,
                content=node.content,
                type=node.node_type.value,
                importance=node.importance,
                entropy=node.entropy,
                label=f"{node.node_type.value}: {node.content[:30]}"
            )
        
        # Add edges
        for edge in graph.edges:
            G.add_edge(edge.source, edge.target, relation=edge.relation, weight=edge.weight)
        
        return G
