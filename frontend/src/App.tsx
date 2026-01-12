import { useState, useEffect } from 'react'
import SemanticTree from './SemanticTree'
import './App.css'

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
  original_tokens: number
}

// Pipeline stages
type PipelineStage = 'idle' | 'building_tree' | 'compressing' | 'decoding' | 'judging' | 'complete'

const API_URL = import.meta.env.DEV ? 'http://localhost:8080' : ''

// Sample message
const SAMPLE_MESSAGE = `URGENT: I've just completed my weekly analysis of the enterprise support queue and we have several critical situations.

CRITICAL - IMMEDIATE ACTION:
1. Ticket #ENG-4521 - GlobalTech Industries (Account: GT-2847) has complete production outage. 2,500 employees affected. Contract guarantees 99.95% uptime with $50,000/hour penalty. Currently at 4.5 hours downtime. CEO Marcus Chen called VP of Sales twice. Root cause: database replication failure after v2.3.1 deployment. On-call engineer James Wilson needs database team support.

2. Ticket #FIN-3892 - Payment processing failing for 847 transactions since 3:47 AM. Total affected: $127,450 across 312 customers. Stripe error PRC-5521 indicates certificate expiration. Finance estimates $89,000 in refunds needed if not resolved in 2 hours.

HIGH PRIORITY - WITHIN 4 HOURS:
3. Tickets #AUTH-2201, #AUTH-2203, #AUTH-2207 - OAuth2 issues causing random logouts after 15 minutes. Affects 23% of mobile users (~45,000 sessions). Yesterday's hotfix made it worse.

4. Ticket #DATA-1156 - Excel exports corrupted for reports >50,000 rows. Fortune 500 clients affected: Walmart #WM-001, Target #TG-445, Costco #CC-892.

ACTIONS NEEDED:
- Escalate #ENG-4521 to database lead Patricia Martinez, loop in VP Engineering
- Create P1 incident report for GlobalTech - executive briefing by 2 PM EST
- Get Stripe team on call within 30 minutes about certificate
- Assign AUTH tickets to identity team, schedule emergency deployment
- Check if DATA-1156 relates to PR #4892 memory optimization

Confirm receipt with ETAs. Monitoring for new critical issues. Direct line: ext. 4455.`

function App() {
  const [message, setMessage] = useState('')
  const [mspEnabled, setMspEnabled] = useState(false)
  const [error, setError] = useState<string | null>(null)
  
  // Pipeline state
  const [stage, setStage] = useState<PipelineStage>('idle')
  const [processing, setProcessing] = useState(false)
  
  // Settings
  const [compressToK, setCompressToK] = useState(15)
  
  // Results
  const [hierarchicalResult, setHierarchicalResult] = useState<HierarchicalResult | null>(null)
  const [decodedText, setDecodedText] = useState<string | null>(null)
  const [similarity, setSimilarity] = useState<number | null>(null)
  const [agentBResponse, setAgentBResponse] = useState<string | null>(null)

  useEffect(() => {
    fetch(`${API_URL}/api/config`)
      .then(res => res.json())
      .then(data => setMspEnabled(data.msp?.enabled ?? false))
      .catch(() => setMspEnabled(false))
  }, [])

  const runPipeline = async () => {
    if (!message.trim()) return
    
    setProcessing(true)
    setError(null)
    setHierarchicalResult(null)
    setDecodedText(null)
    setSimilarity(null)
    setAgentBResponse(null)
    
    try {
      // Step 1: Build hierarchical tree
      setStage('building_tree')
      
      const treeResponse = await fetch(`${API_URL}/api/msp/hierarchical`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ 
          message,
          compress_to_k: compressToK
        })
      })
      
      if (!treeResponse.ok) throw new Error('Failed to build tree')
      const treeData = await treeResponse.json()
      setHierarchicalResult(treeData)
      
      // Step 2: Show compression (already done in tree response)
      setStage('compressing')
      await new Promise(r => setTimeout(r, 500)) // Brief pause to show stage
      
      // Step 3: Decode the compressed signal
      setStage('decoding')
      
      // Build a simple signal from the compressed tree for decoding
      const compressedTree = treeData.compressed_tree || treeData.tree
      const signalForDecode = {
        intent: compressedTree.content,
        entities: compressedTree.children
          .filter((n: TreeNode) => n.level === 'ENTITIES')
          .map((n: TreeNode) => n.content),
        attributes: compressedTree.children
          .filter((n: TreeNode) => n.level === 'ATTRIBUTES')
          .map((n: TreeNode) => n.content),
        details: compressedTree.children
          .filter((n: TreeNode) => n.level === 'DETAILS')
          .map((n: TreeNode) => n.content)
      }
      
      // Use Groq to decode
      const decodeResponse = await fetch(`${API_URL}/api/msp/decode-tree`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ tree_signal: signalForDecode })
      })
      
      let decoded = ''
      if (decodeResponse.ok) {
        const decodeData = await decodeResponse.json()
        decoded = decodeData.decoded_text
        setDecodedText(decoded)
      } else {
        // Fallback: just show the tree content
        decoded = `Intent: ${signalForDecode.intent}\n` +
          `Entities: ${signalForDecode.entities.join(', ')}\n` +
          `Attributes: ${signalForDecode.attributes.join(', ')}`
        setDecodedText(decoded)
      }
      
      // Step 4: Judge similarity
      setStage('judging')
      
      const judgeResponse = await fetch(`${API_URL}/api/msp/judge`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ 
          original: message,
          decoded: decoded
        })
      })
      
      if (judgeResponse.ok) {
        const judgeData = await judgeResponse.json()
        setSimilarity(judgeData.similarity)
      }
      
      // Step 5: Agent B responds
      const agentBResp = await fetch(`${API_URL}/api/msp/agent-respond`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ signal: signalForDecode })
      })
      
      if (agentBResp.ok) {
        const agentBData = await agentBResp.json()
        setAgentBResponse(agentBData.response)
      }
      
      setStage('complete')
      
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Pipeline failed')
      setStage('idle')
    } finally {
      setProcessing(false)
    }
  }

  const loadSample = () => setMessage(SAMPLE_MESSAGE)

  const stageLabels: Record<PipelineStage, string> = {
    idle: 'Ready',
    building_tree: '🌳 Building Hierarchical Tree...',
    compressing: '🗜️ Compressing by Importance...',
    decoding: '📤 Decoding Signal...',
    judging: '⚖️ Judging Similarity...',
    complete: '✅ Complete'
  }

  return (
    <div className="container">
      <header>
        <h1>🔬 Hierarchical Semantic Compression</h1>
        <p className="subtitle">
          Information-theoretic importance scoring for agent communication
        </p>
      </header>

      {/* Pipeline Status */}
      <div className="card pipeline-status">
        <div className="stage-indicator">
          <span className={`stage ${stage !== 'idle' ? 'active' : ''}`}>
            {stageLabels[stage]}
          </span>
          {processing && <span className="spinner">⏳</span>}
        </div>
        
        <div className="pipeline-steps">
          <div className={`step ${stage === 'building_tree' ? 'active' : ['compressing', 'decoding', 'judging', 'complete'].includes(stage) ? 'done' : ''}`}>
            🌳 Tree
          </div>
          <div className="step-arrow">→</div>
          <div className={`step ${stage === 'compressing' ? 'active' : ['decoding', 'judging', 'complete'].includes(stage) ? 'done' : ''}`}>
            🗜️ Compress
          </div>
          <div className="step-arrow">→</div>
          <div className={`step ${stage === 'decoding' ? 'active' : ['judging', 'complete'].includes(stage) ? 'done' : ''}`}>
            📤 Decode
          </div>
          <div className="step-arrow">→</div>
          <div className={`step ${stage === 'judging' ? 'active' : stage === 'complete' ? 'done' : ''}`}>
            ⚖️ Judge
          </div>
          <div className="step-arrow">→</div>
          <div className={`step ${stage === 'complete' ? 'done' : ''}`}>
            🤖 Agent B
          </div>
        </div>
      </div>

      <div className="main-layout">
        {/* Left: Input */}
        <div className="card input-section">
          <h2>📝 Agent A Message</h2>
          <textarea
            value={message}
            onChange={(e) => setMessage(e.target.value)}
            placeholder="Enter agent message..."
            rows={12}
          />
          
          <button className="sample-btn" onClick={loadSample}>
            📋 Load Sample
          </button>
          
          <div className="settings">
            <div className="setting">
              <label>Keep top K nodes</label>
              <input 
                type="range" 
                min="5" max="30" step="1"
                value={compressToK}
                onChange={(e) => setCompressToK(parseInt(e.target.value))}
              />
              <span>{compressToK}</span>
            </div>
          </div>
          
          <button 
            className="run-btn"
            onClick={runPipeline} 
            disabled={processing || !message.trim() || !mspEnabled}
          >
            {processing ? '⏳ Running...' : '🚀 Run Pipeline'}
          </button>
          
          {!mspEnabled && (
            <div className="warning">⚠️ Set GROQ_API_KEY to enable</div>
          )}
          
          {error && <div className="error">❌ {error}</div>}
        </div>

        {/* Right: Results */}
        <div className="results-section">
          {/* Tree Visualization */}
          <div className="card tree-section">
            <h2>🌳 Hierarchical Tree with Importance Scores</h2>
            <p className="description">
              Each node scored by: specificity (rare terms), level (intent &gt; entities &gt; details), 
              numeric content (IDs, amounts), and entropy cost.
            </p>
            
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
          </div>
          
          {/* Metrics */}
          {hierarchicalResult && (
            <div className="card metrics-section">
              <h2>📊 Compression Metrics</h2>
              <div className="metrics-grid">
                <div className="metric">
                  <div className="metric-value">{hierarchicalResult.original_tokens}</div>
                  <div className="metric-label">Original Tokens</div>
                </div>
                <div className="metric">
                  <div className="metric-value">{hierarchicalResult.total_nodes}</div>
                  <div className="metric-label">Tree Nodes</div>
                </div>
                <div className="metric">
                  <div className="metric-value">{hierarchicalResult.compressed_nodes || '-'}</div>
                  <div className="metric-label">After Compression</div>
                </div>
                <div className="metric">
                  <div className="metric-value">
                    {hierarchicalResult.importance_preserved 
                      ? `${(hierarchicalResult.importance_preserved * 100).toFixed(0)}%`
                      : '-'}
                  </div>
                  <div className="metric-label">Importance Kept</div>
                </div>
                <div className="metric">
                  <div className="metric-value">
                    {similarity ? `${(similarity * 100).toFixed(0)}%` : '-'}
                  </div>
                  <div className="metric-label">Semantic Similarity</div>
                </div>
                <div className="metric">
                  <div className="metric-value">{(hierarchicalResult.efficiency * 100).toFixed(0)}%</div>
                  <div className="metric-label">Efficiency</div>
                </div>
              </div>
            </div>
          )}
          
          {/* Decoded Output */}
          {decodedText && (
            <div className="card decoded-section">
              <h2>📤 Decoded from Compressed Signal</h2>
              <div className="decoded-text">{decodedText}</div>
            </div>
          )}
          
          {/* Agent B Response */}
          {agentBResponse && (
            <div className="card agent-b-section">
              <h2>🤖 Agent B Response</h2>
              <div className="agent-response">{agentBResponse}</div>
            </div>
          )}
        </div>
      </div>
    </div>
  )
}

export default App
