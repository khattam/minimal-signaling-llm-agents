import { useState } from 'react'
import './SemanticTree.css'

interface TreeNode {
  content: string
  level: string
  node_type: string
  importance: number
  entropy: number
  children: TreeNode[]
}

interface ParetoPoint {
  target_similarity: number
  minimum_bits: number
  compression_ratio: number
}

interface SemanticTreeProps {
  tree: TreeNode | null
  compressedTree?: TreeNode | null
  paretoFrontier?: ParetoPoint[]
  totalNodes?: number
  totalEntropy?: number
  totalImportance?: number
  compressedNodes?: number
  compressedEntropy?: number
  importancePreserved?: number
}

const LEVEL_COLORS: Record<string, string> = {
  INTENT: '#f97316',
  ENTITIES: '#3b82f6',
  ATTRIBUTES: '#8b5cf6',
  DETAILS: '#6b7280'
}

const LEVEL_EMOJI: Record<string, string> = {
  INTENT: '🎯',
  ENTITIES: '👤',
  ATTRIBUTES: '📊',
  DETAILS: '📝'
}

function TreeNodeComponent({ 
  node, 
  depth = 0,
  isCompressed = false 
}: { 
  node: TreeNode
  depth?: number
  isCompressed?: boolean
}) {
  const [expanded, setExpanded] = useState(depth < 2)
  const hasChildren = node.children && node.children.length > 0
  
  const importanceBar = Math.min(node.importance * 100, 100)
  const color = LEVEL_COLORS[node.level] || '#666'
  const emoji = LEVEL_EMOJI[node.level] || '•'
  
  return (
    <div className={`tree-node ${isCompressed ? 'compressed' : ''}`} style={{ marginLeft: depth * 16 }}>
      <div 
        className="node-header"
        onClick={() => hasChildren && setExpanded(!expanded)}
        style={{ cursor: hasChildren ? 'pointer' : 'default' }}
      >
        {hasChildren && (
          <span className="expand-icon">{expanded ? '▼' : '▶'}</span>
        )}
        <span className="node-emoji">{emoji}</span>
        <span className="node-level" style={{ backgroundColor: color }}>
          {node.level}
        </span>
        <span className="node-type">{node.node_type}</span>
      </div>
      
      <div className="node-content">
        {node.content.length > 60 ? node.content.slice(0, 60) + '...' : node.content}
      </div>
      
      <div className="node-metrics">
        <div className="importance-bar-container">
          <div 
            className="importance-bar"
            style={{ 
              width: `${importanceBar}%`,
              backgroundColor: color
            }}
          />
          <span className="importance-value">{(node.importance * 100).toFixed(1)}%</span>
        </div>
        <span className="entropy-value">{node.entropy.toFixed(1)} bits</span>
      </div>
      
      {expanded && hasChildren && (
        <div className="node-children">
          {node.children.map((child, i) => (
            <TreeNodeComponent 
              key={i} 
              node={child} 
              depth={depth + 1}
              isCompressed={isCompressed}
            />
          ))}
        </div>
      )}
    </div>
  )
}

export default function SemanticTree({
  tree,
  compressedTree,
  paretoFrontier,
  totalNodes,
  totalEntropy,
  totalImportance,
  compressedNodes,
  compressedEntropy,
  importancePreserved
}: SemanticTreeProps) {
  const [showCompressed, setShowCompressed] = useState(false)
  
  if (!tree) {
    return (
      <div className="semantic-tree-placeholder">
        <p>Hierarchical tree will appear here...</p>
      </div>
    )
  }
  
  return (
    <div className="semantic-tree">
      {/* Metrics Summary */}
      <div className="tree-metrics">
        <div className="metric-item">
          <span className="metric-value">{totalNodes}</span>
          <span className="metric-label">Nodes</span>
        </div>
        <div className="metric-item">
          <span className="metric-value">{totalEntropy?.toFixed(0)}</span>
          <span className="metric-label">Bits</span>
        </div>
        <div className="metric-item">
          <span className="metric-value">{((totalImportance || 0) * 100).toFixed(0)}%</span>
          <span className="metric-label">Importance</span>
        </div>
      </div>
      
      {/* Compression Toggle */}
      {compressedTree && (
        <div className="compression-toggle">
          <button 
            className={!showCompressed ? 'active' : ''}
            onClick={() => setShowCompressed(false)}
          >
            Full Tree ({totalNodes} nodes)
          </button>
          <button 
            className={showCompressed ? 'active' : ''}
            onClick={() => setShowCompressed(true)}
          >
            Compressed ({compressedNodes} nodes)
          </button>
        </div>
      )}
      
      {/* Compression Stats */}
      {showCompressed && compressedTree && (
        <div className="compression-stats">
          <span>📉 {compressedEntropy?.toFixed(0)} bits</span>
          <span>✅ {((importancePreserved || 0) * 100).toFixed(0)}% importance kept</span>
        </div>
      )}
      
      {/* Tree Visualization */}
      <div className="tree-container">
        <TreeNodeComponent 
          node={showCompressed && compressedTree ? compressedTree : tree}
          isCompressed={showCompressed}
        />
      </div>
      
      {/* Pareto Frontier */}
      {paretoFrontier && paretoFrontier.length > 0 && (
        <div className="pareto-section">
          <h4>📈 Compression Tradeoff (Pareto Frontier)</h4>
          <p className="pareto-explanation">
            Theoretical minimum bits needed for each similarity level
          </p>
          <div className="pareto-chart">
            {paretoFrontier.map((point, i) => (
              <div key={i} className="pareto-bar">
                <div className="pareto-label">
                  {(point.target_similarity * 100).toFixed(0)}%
                </div>
                <div className="pareto-bar-container">
                  <div 
                    className="pareto-bar-fill"
                    style={{ width: `${point.compression_ratio * 100}%` }}
                  />
                </div>
                <div className="pareto-bits">
                  {point.minimum_bits.toFixed(0)} bits
                </div>
              </div>
            ))}
          </div>
        </div>
      )}
      
      {/* Legend */}
      <div className="tree-legend">
        {Object.entries(LEVEL_COLORS).map(([level, color]) => (
          <span key={level} className="legend-item">
            <span className="legend-dot" style={{ backgroundColor: color }} />
            {LEVEL_EMOJI[level]} {level}
          </span>
        ))}
      </div>
    </div>
  )
}
