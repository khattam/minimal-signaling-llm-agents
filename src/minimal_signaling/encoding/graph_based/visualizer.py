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
        
        # Create pyvis network with better sizing
        net = Network(
            height="900px",
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
              "nodeSpacing": 200,
              "levelSeparation": 250
            }
          },
          "physics": {
            "hierarchicalRepulsion": {
              "centralGravity": 0.0,
              "springLength": 150,
              "springConstant": 0.01,
              "nodeDistance": 150,
              "damping": 0.09
            },
            "solver": "hierarchicalRepulsion",
            "stabilization": {"iterations": 200}
          },
          "nodes": {
            "font": {
              "size": 16,
              "face": "arial"
            }
          }
        }
        """)
        
        # Add nodes with better sizing
        for node_id, node_data in G.nodes(data=True):
            node_type = NodeType(node_data["type"])
            color = self.colors.get(node_type, "#95A5A6")
            
            # Larger size based on importance
            if show_importance:
                size = 20 + (node_data["importance"] * 60)  # 20-80 range
            else:
                size = 40
            
            # Shorter, cleaner label
            content = node_data['content']
            if len(content) > 40:
                label = content[:37] + "..."
            else:
                label = content
            
            # Create title (hover text)
            title_text = f"Type: {node_data['type']}\nContent: {node_data['content']}\nImportance: {node_data['importance']:.3f}\nEntropy: {node_data['entropy']:.2f} bits"
            
            net.add_node(
                node_id,
                label=label,
                title=title_text,
                color=color,
                size=size,
                font={"size": 14, "face": "arial", "color": "#000000"}
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
        
        # Create comparison HTML with legend
        comparison_html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Graph Compression Comparison</title>
            <style>
                body {{ 
                    font-family: 'Segoe UI', Arial, sans-serif; 
                    margin: 20px;
                    background: #f5f5f5;
                }}
                .header {{
                    background: white;
                    padding: 30px;
                    border-radius: 8px;
                    margin-bottom: 20px;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                }}
                h1 {{
                    margin: 0 0 20px 0;
                    color: #333;
                }}
                .stats {{ 
                    background: #f8f9fa; 
                    padding: 20px; 
                    border-radius: 8px; 
                    margin-bottom: 20px;
                    border-left: 4px solid #4ECDC4;
                }}
                .stat {{ 
                    margin: 10px 0;
                    font-size: 16px;
                }}
                .stat-label {{ 
                    font-weight: bold; 
                    color: #555;
                }}
                .legend {{
                    background: white;
                    padding: 20px;
                    border-radius: 8px;
                    margin-bottom: 20px;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                }}
                .legend h2 {{
                    margin-top: 0;
                    color: #333;
                }}
                .legend-items {{
                    display: flex;
                    flex-wrap: wrap;
                    gap: 20px;
                }}
                .legend-item {{
                    display: flex;
                    align-items: center;
                    gap: 10px;
                }}
                .legend-color {{
                    width: 30px;
                    height: 30px;
                    border-radius: 50%;
                    border: 2px solid #ddd;
                }}
                .legend-label {{
                    font-weight: 500;
                    color: #555;
                }}
                .container {{ 
                    display: flex; 
                    gap: 20px;
                    background: white;
                    padding: 20px;
                    border-radius: 8px;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                }}
                .graph {{ 
                    flex: 1;
                }}
                .graph h2 {{
                    margin-top: 0;
                    color: #333;
                    border-bottom: 2px solid #4ECDC4;
                    padding-bottom: 10px;
                }}
                iframe {{ 
                    width: 100%; 
                    height: 700px; 
                    border: 1px solid #ddd;
                    border-radius: 4px;
                    background: white;
                }}
                .note {{
                    background: #fff3cd;
                    border-left: 4px solid #ffc107;
                    padding: 15px;
                    margin-top: 20px;
                    border-radius: 4px;
                }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>🔬 Semantic Graph Compression Analysis</h1>
                
                <div class="stats">
                    <h2>Compression Statistics</h2>
                    <div class="stat">
                        <span class="stat-label">Nodes:</span> 
                        {stats['original_nodes']} → {stats['compressed_nodes']} 
                        ({stats['node_retention']:.1%} retained, {stats['nodes_removed']} removed)
                    </div>
                    <div class="stat">
                        <span class="stat-label">Entropy (Information Content):</span> 
                        {stats['original_entropy']:.2f} → {stats['compressed_entropy']:.2f} bits
                        ({stats['entropy_retention']:.1%} retained)
                    </div>
                    <div class="stat">
                        <span class="stat-label">Importance Score:</span> 
                        {stats['original_importance']:.3f} → {stats['compressed_importance']:.3f}
                        ({stats['importance_retention']:.1%} retained)
                    </div>
                </div>
                
                <div class="legend">
                    <h2>Node Type Legend</h2>
                    <div class="legend-items">
                        <div class="legend-item">
                            <div class="legend-color" style="background: #FF6B6B;"></div>
                            <span class="legend-label">INTENT - Actions requested</span>
                        </div>
                        <div class="legend-item">
                            <div class="legend-color" style="background: #4ECDC4;"></div>
                            <span class="legend-label">ENTITY - Who/what is involved</span>
                        </div>
                        <div class="legend-item">
                            <div class="legend-color" style="background: #45B7D1;"></div>
                            <span class="legend-label">ATTRIBUTE - Properties, numbers, dates</span>
                        </div>
                        <div class="legend-item">
                            <div class="legend-color" style="background: #96CEB4;"></div>
                            <span class="legend-label">DETAIL - Context and explanations</span>
                        </div>
                        <div class="legend-item">
                            <div class="legend-color" style="background: #FFEAA7;"></div>
                            <span class="legend-label">CONSTRAINT - Requirements, limits</span>
                        </div>
                        <div class="legend-item">
                            <div class="legend-color" style="background: #DFE6E9;"></div>
                            <span class="legend-label">OUTCOME - Expected results</span>
                        </div>
                    </div>
                </div>
                
                <div class="note">
                    <strong>💡 How to use:</strong> Hover over nodes to see full content. Larger nodes = higher importance. 
                    Arrows show relationships between concepts. Compression removes low-importance nodes while preserving structure.
                </div>
            </div>
            
            <div class="container">
                <div class="graph">
                    <h2>📊 Original Graph ({stats['original_nodes']} nodes)</h2>
                    <iframe src="graph_original.html"></iframe>
                </div>
                <div class="graph">
                    <h2>🗜️ Compressed Graph ({stats['compressed_nodes']} nodes)</h2>
                    <iframe src="graph_compressed.html"></iframe>
                </div>
            </div>
        </body>
        </html>
        """
        
        (output_dir / "comparison.html").write_text(comparison_html, encoding='utf-8')
        print(f"Comparison saved to: {output_dir / 'comparison.html'}")
