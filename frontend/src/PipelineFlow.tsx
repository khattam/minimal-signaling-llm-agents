import { useCallback } from 'react'
import ReactFlow, {
  type Node,
  type Edge,
  Background,
  MarkerType,
  Position,
} from 'reactflow'
import 'reactflow/dist/style.css'

export type PipelineStage = 
  | 'idle'
  | 'encoding'
  | 'decoding'
  | 'judging'
  | 'analyzing'
  | 'refining'
  | 'agent_b'
  | 'complete'
  | 'error'

interface PipelineFlowProps {
  currentStage: PipelineStage
  iteration: number
  similarity: number | null
  targetSimilarity: number
}

const nodeStyle = {
  padding: '10px 20px',
  borderRadius: '8px',
  fontSize: '12px',
  fontWeight: 500,
  border: '2px solid #333',
  background: '#1a1a2e',
  color: '#fff',
  minWidth: '100px',
  textAlign: 'center' as const,
}

const activeStyle = {
  ...nodeStyle,
  border: '2px solid #22c55e',
  boxShadow: '0 0 20px rgba(34, 197, 94, 0.5)',
  background: '#1a3a2e',
}

const completedStyle = {
  ...nodeStyle,
  border: '2px solid #3b82f6',
  background: '#1a2a3e',
}

export default function PipelineFlow({ 
  currentStage, 
  iteration, 
  similarity,
  targetSimilarity 
}: PipelineFlowProps) {
  const getNodeStyle = useCallback((nodeId: string) => {
    const stageMap: Record<string, string[]> = {
      'agent_a': ['encoding'],
      'encoder': ['encoding', 'refining'],
      'decoder': ['decoding'],
      'judge': ['judging'],
      'analyzer': ['analyzing'],
      'agent_b': ['agent_b'],
    }
    
    if (currentStage === 'complete') return completedStyle
    if (currentStage === 'idle') return nodeStyle
    if (stageMap[nodeId]?.includes(currentStage)) return activeStyle
    return nodeStyle
  }, [currentStage])

  const nodes: Node[] = [
    {
      id: 'agent_a',
      position: { x: 50, y: 150 },
      data: { label: '🤖 Agent A\n(Input)' },
      style: getNodeStyle('agent_a'),
      sourcePosition: Position.Right,
    },
    {
      id: 'encoder',
      position: { x: 200, y: 150 },
      data: { 
        label: currentStage === 'refining' 
          ? `🔄 Encoder\n(Iter ${iteration})` 
          : '📝 Encoder\n(NL → MSP)' 
      },
      style: getNodeStyle('encoder'),
      sourcePosition: Position.Right,
      targetPosition: Position.Left,
    },
    {
      id: 'decoder',
      position: { x: 350, y: 150 },
      data: { label: '📖 Decoder\n(MSP → NL)' },
      style: getNodeStyle('decoder'),
      sourcePosition: Position.Right,
      targetPosition: Position.Left,
    },
    {
      id: 'judge',
      position: { x: 500, y: 150 },
      data: { 
        label: similarity !== null 
          ? `⚖️ Judge\n${(similarity * 100).toFixed(0)}% sim` 
          : '⚖️ Judge\n(Semantic)' 
      },
      style: getNodeStyle('judge'),
      sourcePosition: Position.Right,
      targetPosition: Position.Left,
    },
    {
      id: 'analyzer',
      position: { x: 350, y: 50 },
      data: { label: '🔍 Analyzer\n(Loss Detection)' },
      style: getNodeStyle('analyzer'),
      sourcePosition: Position.Left,
      targetPosition: Position.Right,
    },
    {
      id: 'agent_b',
      position: { x: 650, y: 150 },
      data: { label: '🤖 Agent B\n(Output)' },
      style: getNodeStyle('agent_b'),
      targetPosition: Position.Left,
    },
  ]

  const edges: Edge[] = [
    {
      id: 'e1',
      source: 'agent_a',
      target: 'encoder',
      animated: currentStage === 'encoding',
      style: { stroke: '#666' },
      markerEnd: { type: MarkerType.ArrowClosed, color: '#666' },
    },
    {
      id: 'e2',
      source: 'encoder',
      target: 'decoder',
      animated: currentStage === 'decoding',
      style: { stroke: '#666' },
      markerEnd: { type: MarkerType.ArrowClosed, color: '#666' },
    },
    {
      id: 'e3',
      source: 'decoder',
      target: 'judge',
      animated: currentStage === 'judging',
      style: { stroke: '#666' },
      markerEnd: { type: MarkerType.ArrowClosed, color: '#666' },
    },
    {
      id: 'e4',
      source: 'judge',
      target: 'analyzer',
      label: similarity !== null && similarity < targetSimilarity ? 'below threshold' : '',
      animated: currentStage === 'analyzing',
      style: { stroke: '#f59e0b' },
      markerEnd: { type: MarkerType.ArrowClosed, color: '#f59e0b' },
      labelStyle: { fill: '#f59e0b', fontSize: 10 },
    },
    {
      id: 'e5',
      source: 'analyzer',
      target: 'encoder',
      label: 'feedback',
      animated: currentStage === 'refining',
      style: { stroke: '#f59e0b' },
      markerEnd: { type: MarkerType.ArrowClosed, color: '#f59e0b' },
      labelStyle: { fill: '#f59e0b', fontSize: 10 },
    },
    {
      id: 'e6',
      source: 'judge',
      target: 'agent_b',
      label: similarity !== null && similarity >= targetSimilarity ? '✓ passed' : '',
      animated: currentStage === 'agent_b',
      style: { stroke: '#22c55e' },
      markerEnd: { type: MarkerType.ArrowClosed, color: '#22c55e' },
      labelStyle: { fill: '#22c55e', fontSize: 10 },
    },
  ]

  return (
    <div style={{ width: '100%', height: '250px', background: '#0d0d1a', borderRadius: '8px' }}>
      <ReactFlow
        nodes={nodes}
        edges={edges}
        fitView
        attributionPosition="bottom-left"
        proOptions={{ hideAttribution: true }}
        nodesDraggable={false}
        nodesConnectable={false}
        elementsSelectable={false}
        zoomOnScroll={false}
        panOnScroll={false}
        panOnDrag={false}
      >
        <Background color="#333" gap={20} />
      </ReactFlow>
    </div>
  )
}
