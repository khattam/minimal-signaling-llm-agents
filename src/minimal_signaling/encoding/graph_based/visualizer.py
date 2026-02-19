"""Graph Visualizer - Creates interactive visualizations of semantic graphs."""

from pathlib import Path
from typing import Optional
import networkx as nx
from pyvis.network import Network

from .semantic_graph import SemanticGraph, NodeType
from .graph_compressor import GraphCompressor


class GraphVisualizer:
    """Creates interactive visualizations of semantic graphs."""
    
    def __init__(self):
        self.compressor = GraphCompressor()
        
        # Color scheme for node types
        self.colors = {
            NodeType.INTENT: "#FF6B6B",      # Red
            NodeType.ENTITY: "#4ECDC4",      # Teal
            NodeType.ATTRIBUTE: "#45B7D1",   # Blue
            NodeType.DETAIL: "#96CEB4",      # Green
            NodeType.CONSTRAINT: "#FFEAA7",  # Yellow
            NodeType.OUTCOME: "#DFE6E9"      # Gray
        }
    
    def visualize(
        self,
        graph: SemanticGraph,
        output_path: str = "graph_viz.html",
        title: str = "Semantic Graph",
        show_importance: bool = True
    ):
        """Create interactive HTML visualization of the graph.
        
        Args:
            graph: Semantic graph to visualize
            output_path: Path to save HTML file
            title: Title for the visualization
            show_importance: Whether to size nodes by importance
        """
        # Convert to NetworkX
        G = self.compressor.to_networkx(graph)
        
        # Create pyvis network
        net = Network(
            height="750px",
            width="100%",
            bgcolor="#ffffff",
            font_color="#000000",
            directed=True,
            notebook=False
        )
        
        # Configure physics for hierarchical layout
        net.set_options("""
        {
          "layout": {
            "hierarchical": {
              "enabled": true,
              "direction": "UD",
              "sortMethod": "directed",
              "nodeSpacing": 150,
              "levelSeparation": 200
            }
          },
          "physics": {
            "hierarchicalRepulsion": {
              "centralGravity": 0.0,
              "springLength": 100,
              "springConstant": 0.01,
              "nodeDistance": 120,
              "damping": 0.09
            },
            "solver": "hierarchicalRepulsion",
            "stabilization": {"iterations": 200}
          }
        }
        """)
        
        # Add nodes
        for node_id, node_data in G.nodes(data=True):
            node_type = NodeType(node_data["type"])
            color = self.colors.get(node_type, "#95A5A6")
            
            # Size based on importance if enabled
            if show_importance:
                size = 10 + (node_data["importance"] * 40)  # 10-50 range
            else:
                size = 25
            
            # Create label
            label = f"{node_data['type']}\n{node_data['content'][:50]}"
            if len(node_data['content']) > 50:
                label += "..."
            
            # Create title (hover text)
            title_text = f"""
            Type: {node_data['type']}
            Content: {node_data['content']}
            Importance: {node_data['importance']:.3f}
            Entropy: {node_data['entropy']:.2f} bits
            """
            
            net.add_node(
                node_id,
                label=label,
                title=title_text,
                color=color,
                size=size,
                font={"size": 12}
            )
        
        # Add edges
        for source, target, edge_data in G.edges(data=True):
            net.add_edge(
                source,
                target,
                title=edge_data.get("relation", "related_to"),
                arrows="to"
            )
        
        # Save with proper parameters
        try:
            net.save_graph(output_path)
            print(f"Visualization saved to: {output_path}")
        except Exception as e:
            print(f"Warning: Could not create visualization: {e}")
    
    def visualize_comparison(
        self,
        original: SemanticGraph,
        compressed: SemanticGraph,
        output_dir: str = "."
    ):
        """Create side-by-side visualizations of original and compressed graphs.
        
        Args:
            original: Original semantic graph
            compressed: Compressed semantic graph
            output_dir: Directory to save HTML files
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        
        # Visualize original
        self.visualize(
            original,
            str(output_dir / "graph_original.html"),
            "Original Semantic Graph"
        )
        
        # Visualize compressed
        self.visualize(
            compressed,
            str(output_dir / "graph_compressed.html"),
            "Compressed Semantic Graph"
        )
        
        # Get stats
        stats = self.compressor.get_compression_stats(original, compressed)
        
        # Create comparison HTML
        comparison_html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Graph Compression Comparison</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .container {{ display: flex; gap: 20px; }}
                .graph {{ flex: 1; }}
                .stats {{ background: #f0f0f0; padding: 20px; border-radius: 8px; margin-bottom: 20px; }}
                .stat {{ margin: 10px 0; }}
                .stat-label {{ font-weight: bold; }}
                iframe {{ width: 100%; height: 600px; border: 1px solid #ccc; }}
            </style>
        </head>
        <body>
            <h1>Semantic Graph Compression Comparison</h1>
            
            <div class="stats">
                <h2>Compression Statistics</h2>
                <div class="stat">
                    <span class="stat-label">Nodes:</span> 
                    {stats['original_nodes']} → {stats['compressed_nodes']} 
                    ({stats['node_retention']:.1%} retained)
                </div>
                <div class="stat">
                    <span class="stat-label">Entropy:</span> 
                    {stats['original_entropy']:.2f} → {stats['compressed_entropy']:.2f} bits
                    ({stats['entropy_retention']:.1%} retained)
                </div>
                <div class="stat">
                    <span class="stat-label">Importance:</span> 
                    {stats['original_importance']:.3f} → {stats['compressed_importance']:.3f}
                    ({stats['importance_retention']:.1%} retained)
                </div>
            </div>
            
            <div class="container">
                <div class="graph">
                    <h2>Original Graph</h2>
                    <iframe src="graph_original.html"></iframe>
                </div>
                <div class="graph">
                    <h2>Compressed Graph</h2>
                    <iframe src="graph_compressed.html"></iframe>
                </div>
            </div>
        </body>
        </html>
        """
        
        (output_dir / "comparison.html").write_text(comparison_html, encoding='utf-8')
        print(f"Comparison saved to: {output_dir / 'comparison.html'}")
