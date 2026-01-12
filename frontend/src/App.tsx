import { useState, useEffect } from 'react'
import PipelineFlow, { type PipelineStage } from './PipelineFlow'
import SemanticTree from './SemanticTree'
import './App.css'

interface MSPSignal {
  version: string
  intent: string
  target: string
  params: Record<string, unknown>
  constraints: string[]
  state: Record<string, unknown>
  priority: string
  trace_id: string
  timestamp: string
}

interface RefinementStep {
  iteration: number
  signal_tokens: number
  similarity: number
  feedback: string | null
  intent: string
  target: string
}

interface IterativeResult {
  success: boolean
  agent_a_message: string
  agent_a_tokens: number
  iterations: number
  converged: boolean
  refinement_history: RefinementStep[]
  final_signal: MSPSignal
  final_signal_json: string
  final_signal_tokens: number
  final_similarity: number
  agent_b_response: string
  agent_b_tokens: number
  compression_ratio: number
  tokens_saved: number
  latency_ms: number
}

interface StreamEvent {
  stage: PipelineStage
  iteration?: number
  similarity?: number
  passed_threshold?: boolean
  feedback?: string
  result?: IterativeResult
  error?: string
}

// Hierarchical tree types
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

interface HierarchicalResult {
  success: boolean
  tree: TreeNode
  total_nodes: number
  total_entropy: number
  total_importance: number
  pareto_frontier: ParetoPoint[]
  theoretical_bound_80: number
  efficiency: number
  compressed_tree?: TreeNode
  compressed_nodes?: number
  compressed_entropy?: number
  importance_preserved?: number
  latency_ms: number
}

const API_URL = import.meta.env.DEV ? 'http://localhost:8080' : ''

// Sample message for testing
const SAMPLE_MESSAGE = `I've completed my analysis of the customer support ticket backlog and need to coordinate with the escalation team. Here's what I found:

URGENT ISSUES (require immediate attention):
1. Ticket #4521 - Enterprise client Acme Corp experiencing complete service outage for 6+ hours. Their SLA guarantees 99.9% uptime and we're now in breach. They're threatening contract termination worth $2.3M annually. The root cause appears to be a misconfigured load balancer after last night's deployment.

2. Ticket #4518 - Payment processing failures affecting approximately 340 transactions since 2:00 AM EST. Customers are being charged but orders aren't completing. Finance team estimates $47,000 in pending refunds needed.

HIGH PRIORITY (within 24 hours):
3. Tickets #4502, #4507, #4511 - All related to the new authentication flow. Users report being logged out randomly mid-session. Affects roughly 12% of active users based on error logs.

4. Ticket #4499 - Data export feature returning corrupted CSV files for reports over 10,000 rows. Three enterprise clients have reported this.

I need you to:
- Immediately escalate tickets #4521 and #4518 to the on-call engineering team
- Create an incident report for the Acme Corp situation for executive review
- Group the authentication tickets and assign to the identity team
- Verify if the CSV issue is related to the recent database migration

Please confirm receipt and provide ETAs for each action item. I'll continue monitoring incoming tickets and flag anything else critical.`

// Intent descriptions
const INTENT_INFO: Record<string, string> = {
  DELEGATE: 'Asking another agent to perform tasks',
  ANALYZE: 'Request to examine or investigate',
  GENERATE: 'Create new content or output',
  EVALUATE: 'Assess, judge, or review something',
  TRANSFORM: 'Convert or modify data/format',
  QUERY: 'Ask a question or request info',
  RESPOND: 'Reply to a previous message',
  REPORT: 'Provide status or findings'
}

function App() {
  const [message, setMessage] = useState('')
  const [result, setResult] = useState<IterativeResult | null>(null)
  const [processing, setProcessing] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [mspEnabled, setMspEnabled] = useState(false)
  const [targetSimilarity, setTargetSimilarity] = useState(0.80)
  const [maxIterations, setMaxIterations] = useState(3)
  const [showExplainer, setShowExplainer] = useState(false)
  const [expandedFeedback, setExpandedFeedback] = useState<number | null>(null)
  
  // Real-time pipeline state
  const [pipelineStage, setPipelineStage] = useState<PipelineStage>('idle')
  const [currentIteration, setCurrentIteration] = useState(0)
  const [currentSimilarity, setCurrentSimilarity] = useState<number | null>(null)
  const [eventLog, setEventLog] = useState<string[]>([])
  
  // Hierarchical tree state
  const [hierarchicalResult, setHierarchicalResult] = useState<HierarchicalResult | null>(null)
  const [compressToK, setCompressToK] = useState(10)
  const [loadingTree, setLoadingTree] = useState(false)
  const [activeTab, setActiveTab] = useState<'pipeline' | 'tree'>('pipeline')

  useEffect(() => {
    fetch(`${API_URL}/api/config`)
      .then(res => res.json())
      .then(data => setMspEnabled(data.msp?.enabled ?? false))
      .catch(() => setMspEnabled(false))
  }, [])

  const processMessage = async () => {
    if (!message.trim()) return
    
    setProcessing(true)
    setResult(null)
    setError(null)
    setPipelineStage('idle')
    setCurrentIteration(0)
    setCurrentSimilarity(null)
    setEventLog([])
    setExpandedFeedback(null)
    
    try {
      const response = await fetch(`${API_URL}/api/msp/iterative-flow-stream`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ 
          agent_a_message: message,
          target_similarity: targetSimilarity,
          max_iterations: maxIterations
        })
      })
      
      if (!response.ok) {
        throw new Error('Stream request failed')
      }
      
      const reader = response.body?.getReader()
      if (!reader) throw new Error('No reader available')
      
      const decoder = new TextDecoder()
      let buffer = ''
      
      while (true) {
        const { done, value } = await reader.read()
        if (done) break
        
        buffer += decoder.decode(value, { stream: true })
        const lines = buffer.split('\n')
        buffer = lines.pop() || ''
        
        for (const line of lines) {
          if (line.startsWith('data: ')) {
            try {
              const event: StreamEvent = JSON.parse(line.slice(6))
              
              if (event.stage) {
                setPipelineStage(event.stage)
                setEventLog(prev => [...prev, `${event.stage} (iter ${event.iteration || '-'})`])
              }
              if (event.iteration) {
                setCurrentIteration(event.iteration)
              }
              if (event.similarity !== undefined && event.similarity !== null) {
                setCurrentSimilarity(event.similarity)
              }
              
              if (event.stage === 'complete' && event.result) {
                setResult(event.result)
              }
              
              if (event.stage === 'error') {
                setError(event.error || 'Unknown error')
              }
            } catch (e) {
              console.error('Failed to parse SSE event:', e)
            }
          }
        }
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Processing failed')
      setPipelineStage('idle')
    } finally {
      setProcessing(false)
    }
  }

  const loadSample = () => {
    setMessage(SAMPLE_MESSAGE)
  }

  const fetchHierarchicalTree = async () => {
    if (!message.trim()) return
    
    setLoadingTree(true)
    setError(null)
    
    try {
      const response = await fetch(`${API_URL}/api/msp/hierarchical`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ 
          message,
          compress_to_k: compressToK
        })
      })
      
      if (!response.ok) {
        throw new Error('Hierarchical encoding failed')
      }
      
      const data = await response.json()
      setHierarchicalResult(data)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to encode')
    } finally {
      setLoadingTree(false)
    }
  }

  // Parse feedback into structured format
  const parseFeedback = (feedback: string) => {
    const lines = feedback.split('\n').filter(l => l.trim())
    const issues: { type: string; items: string[] }[] = []
    let currentType = 'Issues'
    let currentItems: string[] = []
    
    for (const line of lines) {
      const lower = line.toLowerCase()
      if (lower.includes('missing') || lower.includes('lost')) {
        if (currentItems.length) issues.push({ type: currentType, items: currentItems })
        currentType = '❌ Missing'
        currentItems = []
      } else if (lower.includes('distort') || lower.includes('changed')) {
        if (currentItems.length) issues.push({ type: currentType, items: currentItems })
        currentType = '⚠️ Distorted'
        currentItems = []
      } else if (lower.includes('oversimplif') || lower.includes('simplified')) {
        if (currentItems.length) issues.push({ type: currentType, items: currentItems })
        currentType = '📉 Oversimplified'
        currentItems = []
      } else if (line.trim().startsWith('-') || line.trim().startsWith('*') || line.trim().match(/^\d+\./)) {
        currentItems.push(line.replace(/^[-*\d.]+\s*/, '').trim())
      } else if (line.trim()) {
        currentItems.push(line.trim())
      }
    }
    if (currentItems.length) issues.push({ type: currentType, items: currentItems })
    
    return issues
  }

  const intentColors: Record<string, string> = {
    ANALYZE: '#f59e0b', GENERATE: '#22c55e', EVALUATE: '#38bdf8',
    TRANSFORM: '#a855f7', QUERY: '#ec4899', RESPOND: '#14b8a6',
    DELEGATE: '#f97316', REPORT: '#6366f1'
  }

  return (
    <div className="container">
      <header>
        <h1>🔬 MSP with Iterative Refinement</h1>
        <p className="subtitle">
          {mspEnabled 
            ? 'Novel: Semantic feedback loop for optimal compression'
            : '⚠️ Set GROQ_API_KEY to enable'}
        </p>
        <button 
          className="explainer-toggle"
          onClick={() => setShowExplainer(!showExplainer)}
        >
          {showExplainer ? '✕ Hide' : '❓ How it works'}
        </button>
      </header>

      {/* Explainer Panel */}
      {showExplainer && (
        <div className="card explainer-card">
          <h2>🧠 How the System Works</h2>
          <div className="explainer-grid">
            <div className="explainer-section">
              <h3>📊 Token Counting</h3>
              <p>Uses <code>tiktoken</code> (OpenAI's tokenizer). 
              Tokens ≈ 4 chars or 0.75 words. We measure compression by comparing 
              original message tokens vs MSP signal JSON tokens.</p>
            </div>
            <div className="explainer-section">
              <h3>🎯 Semantic Similarity</h3>
              <p>Uses <code>sentence-transformers</code> (all-MiniLM-L6-v2). 
              Converts text → 384D vectors, then cosine similarity:</p>
              <code className="formula">sim = (A · B) / (||A|| × ||B||)</code>
            </div>
            <div className="explainer-section">
              <h3>🔄 Feedback Loop</h3>
              <p>When similarity &lt; threshold:</p>
              <ol>
                <li>LLM analyzes what info was lost</li>
                <li>Feedback injected into encoder</li>
                <li>Re-encode with focus on missing info</li>
              </ol>
            </div>
            <div className="explainer-section">
              <h3>🏷️ Intent Types</h3>
              <div className="intent-list">
                {Object.entries(INTENT_INFO).map(([intent, desc]) => (
                  <div key={intent} className="intent-item">
                    <span className="intent-badge" style={{ backgroundColor: intentColors[intent] }}>{intent}</span>
                    <span>{desc}</span>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Tab Navigation */}
      <div className="view-tabs">
        <button 
          className={activeTab === 'pipeline' ? 'active' : ''}
          onClick={() => setActiveTab('pipeline')}
        >
          🔄 Iterative Pipeline
        </button>
        <button 
          className={activeTab === 'tree' ? 'active' : ''}
          onClick={() => setActiveTab('tree')}
        >
          🌳 Hierarchical Tree
        </button>
      </div>

      {/* Live Pipeline Visualization */}
      {activeTab === 'pipeline' && (
      <div className="card pipeline-card">
        <h2>🔌 Live Pipeline {processing && <span className="live-indicator">● LIVE</span>}</h2>
        <PipelineFlow 
          currentStage={pipelineStage}
          iteration={currentIteration}
          similarity={currentSimilarity}
          targetSimilarity={targetSimilarity}
        />
        <div className="pipeline-legend">
          <span><span className="legend-dot active"></span> Active</span>
          <span><span className="legend-dot feedback"></span> Feedback Loop</span>
          <span><span className="legend-dot complete"></span> Complete</span>
        </div>
        {eventLog.length > 0 && (
          <div className="event-log">
            {eventLog.map((e, i) => <span key={i}>{e}</span>)}
          </div>
        )}
      </div>
      )}

      {/* Hierarchical Tree View */}
      {activeTab === 'tree' && (
        <div className="card tree-card">
          <h2>🌳 Hierarchical Semantic Tree</h2>
          <p className="tree-description">
            Information-theoretic importance scoring with compression bounds
          </p>
          
          <div className="tree-controls">
            <div className="setting">
              <label>Compress to top K nodes</label>
              <input 
                type="range" 
                min="3" max="30" step="1"
                value={compressToK}
                onChange={(e) => setCompressToK(parseInt(e.target.value))}
              />
              <span>{compressToK}</span>
            </div>
            <button 
              onClick={fetchHierarchicalTree} 
              disabled={loadingTree || !message.trim() || !mspEnabled}
            >
              {loadingTree ? '⏳ Encoding...' : '🌳 Build Tree'}
            </button>
          </div>
          
          <SemanticTree 
            tree={hierarchicalResult?.tree || null}
            compressedTree={hierarchicalResult?.compressed_tree}
            paretoFrontier={hierarchicalResult?.pareto_frontier}
            totalNodes={hierarchicalResult?.total_nodes}
            totalEntropy={hierarchicalResult?.total_entropy}
            totalImportance={hierarchicalResult?.total_importance}
            compressedNodes={hierarchicalResult?.compressed_nodes}
            compressedEntropy={hierarchicalResult?.compressed_entropy}
            importancePreserved={hierarchicalResult?.importance_preserved}
          />
          
          {hierarchicalResult && (
            <div className="tree-metrics-bar">
              <span>⏱️ {(hierarchicalResult.latency_ms / 1000).toFixed(1)}s</span>
              <span>📐 Efficiency: {(hierarchicalResult.efficiency * 100).toFixed(0)}%</span>
              <span>🎯 80% sim needs: {hierarchicalResult.theoretical_bound_80.toFixed(0)} bits</span>
            </div>
          )}
        </div>
      )}

      <div className="main-grid">
        {/* Left: Input */}
        <div className="card input-card">
          <h2>🤖 Agent A Message</h2>
          <textarea
            value={message}
            onChange={(e) => setMessage(e.target.value)}
            placeholder="Enter a complex agent message..."
          />
          
          <button className="sample-btn" onClick={loadSample}>
            📋 Load Sample Message
          </button>
          
          <div className="settings">
            <div className="setting">
              <label>Target Similarity</label>
              <input 
                type="range" 
                min="0.6" max="0.95" step="0.05"
                value={targetSimilarity}
                onChange={(e) => setTargetSimilarity(parseFloat(e.target.value))}
              />
              <span>{(targetSimilarity * 100).toFixed(0)}%</span>
            </div>
            <div className="setting">
              <label>Max Iterations</label>
              <input 
                type="range" 
                min="1" max="5" step="1"
                value={maxIterations}
                onChange={(e) => setMaxIterations(parseInt(e.target.value))}
              />
              <span>{maxIterations}</span>
            </div>
          </div>
          
          <button onClick={processMessage} disabled={processing || !message.trim() || !mspEnabled}>
            {processing ? '⏳ Processing...' : '🚀 Encode with Refinement'}
          </button>
          
          {error && <div className="error">❌ {error}</div>}
        </div>

        {/* Middle: Refinement History */}
        <div className="card refinement-card">
          <h2>🔄 Refinement Iterations</h2>
          {result ? (
            <div className="iterations">
              {result.refinement_history.map((step, i) => (
                <div key={i} className={`iteration ${i === result.iterations - 1 ? 'final' : ''}`}>
                  <div className="iter-header">
                    <span className="iter-num">#{step.iteration}</span>
                    <span 
                      className="iter-intent"
                      style={{ backgroundColor: intentColors[step.intent] || '#666' }}
                      title={INTENT_INFO[step.intent]}
                    >
                      {step.intent}
                    </span>
                  </div>
                  <div className="iter-target">{step.target}</div>
                  <div className="iter-metrics">
                    <span>{step.signal_tokens} tokens</span>
                    <span className={step.similarity >= targetSimilarity ? 'good' : 'bad'}>
                      {(step.similarity * 100).toFixed(1)}% sim
                    </span>
                  </div>
                  <div className="similarity-bar">
                    <div 
                      className="similarity-fill"
                      style={{ 
                        width: `${step.similarity * 100}%`,
                        backgroundColor: step.similarity >= targetSimilarity ? '#22c55e' : '#f59e0b'
                      }}
                    />
                    <div 
                      className="threshold-line"
                      style={{ left: `${targetSimilarity * 100}%` }}
                    />
                  </div>
                  {step.feedback && (
                    <div className="feedback-section">
                      <button 
                        className="feedback-toggle"
                        onClick={() => setExpandedFeedback(expandedFeedback === i ? null : i)}
                      >
                        💡 {expandedFeedback === i ? 'Hide' : 'Show'} Loss Analysis
                      </button>
                      {expandedFeedback === i && (
                        <div className="feedback-details">
                          {parseFeedback(step.feedback).map((group, gi) => (
                            <div key={gi} className="feedback-group">
                              <div className="feedback-type">{group.type}</div>
                              <ul>
                                {group.items.map((item, ii) => (
                                  <li key={ii}>{item}</li>
                                ))}
                              </ul>
                            </div>
                          ))}
                        </div>
                      )}
                    </div>
                  )}
                </div>
              ))}
              
              <div className="convergence-status">
                {result.converged 
                  ? '✅ Converged!' 
                  : `⚠️ Max iterations (${result.iterations}) reached`}
              </div>
            </div>
          ) : (
            <div className="placeholder">
              {processing ? 'Processing...' : 'Refinement iterations will appear here...'}
            </div>
          )}
        </div>

        {/* Right: Final Signal & Agent B */}
        <div className="right-column">
          {result && (
            <>
              <div className="card signal-card">
                <h2>📦 Final MSP Signal</h2>
                <div className="signal-json">
                  <pre>{result.final_signal_json}</pre>
                </div>
              </div>
              
              <div className="card response-card">
                <h2>🤖 Agent B Response</h2>
                <div className="agent-response">
                  {result.agent_b_response}
                </div>
              </div>
            </>
          )}
        </div>
      </div>

      {/* Metrics Bar */}
      {result && (
        <div className="card metrics-card">
          <div className="metrics-grid">
            <div className="metric">
              <div className="metric-value">{result.agent_a_tokens}</div>
              <div className="metric-label">Original Tokens</div>
            </div>
            <div className="metric">
              <div className="metric-value highlight">{result.final_signal_tokens}</div>
              <div className="metric-label">Signal Tokens</div>
            </div>
            <div className="metric">
              <div className="metric-value success">
                {result.tokens_saved > 0 ? '-' : '+'}{Math.abs(result.tokens_saved)}
              </div>
              <div className="metric-label">Tokens Saved</div>
            </div>
            <div className="metric">
              <div className="metric-value">{result.iterations}</div>
              <div className="metric-label">Iterations</div>
            </div>
            <div className="metric">
              <div className="metric-value">{(result.final_similarity * 100).toFixed(0)}%</div>
              <div className="metric-label">Final Similarity</div>
            </div>
            <div className="metric">
              <div className="metric-value">{(result.latency_ms / 1000).toFixed(1)}s</div>
              <div className="metric-label">Total Time</div>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}

export default App
