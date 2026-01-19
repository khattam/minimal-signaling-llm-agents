import { useState, useEffect } from 'react'
import PipelineFlow, { type PipelineStage } from './PipelineFlow'
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

const API_URL = import.meta.env.DEV ? 'http://localhost:8080' : ''

const SAMPLE_MESSAGE = `I've completed my analysis of the customer support ticket backlog and need to coordinate with the escalation team.

URGENT ISSUES:
1. Ticket #4521 - Enterprise client Acme Corp experiencing complete service outage for 6+ hours. SLA breach imminent, contract worth $2.3M annually. Root cause: misconfigured load balancer after deployment.

2. Ticket #4518 - Payment processing failures affecting 340 transactions since 2:00 AM. Customers charged but orders not completing. Estimated $47,000 in pending refunds.

HIGH PRIORITY:
3. Tickets #4502, #4507, #4511 - Authentication flow issues causing random logouts. Affects 12% of active users.

4. Ticket #4499 - Data export returning corrupted CSV files for reports over 10,000 rows.

Actions needed:
- Escalate #4521 and #4518 to on-call engineering
- Create incident report for Acme Corp
- Group authentication tickets for identity team
- Check if CSV issue relates to database migration

Please confirm receipt and provide ETAs.`

function App() {
  const [message, setMessage] = useState('')
  const [result, setResult] = useState<IterativeResult | null>(null)
  const [processing, setProcessing] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [mspEnabled, setMspEnabled] = useState(false)
  const [targetSimilarity, setTargetSimilarity] = useState(0.80)
  const [maxIterations, setMaxIterations] = useState(3)
  const [expandedFeedback, setExpandedFeedback] = useState<number | null>(null)
  
  const [pipelineStage, setPipelineStage] = useState<PipelineStage>('idle')
  const [currentIteration, setCurrentIteration] = useState(0)
  const [currentSimilarity, setCurrentSimilarity] = useState<number | null>(null)

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
      
      if (!response.ok) throw new Error('Request failed')
      
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
              
              if (event.stage) setPipelineStage(event.stage)
              if (event.iteration) setCurrentIteration(event.iteration)
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
              console.error('Parse error:', e)
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

  const loadSample = () => setMessage(SAMPLE_MESSAGE)

  return (
    <div className="container">
      <header>
        <h1>Semantic Compression Pipeline</h1>
        <p className="subtitle">Iterative refinement with feedback loop</p>
      </header>

      {!mspEnabled && (
        <div className="warning-banner">
          GROQ_API_KEY not configured. Set it in .env file to enable processing.
        </div>
      )}

      {/* Pipeline Visualization */}
      <div className="card">
        <div className="card-header">
          <h2>Pipeline Status</h2>
          {processing && <span className="status-badge processing">Processing</span>}
        </div>
        <PipelineFlow 
          currentStage={pipelineStage}
          iteration={currentIteration}
          similarity={currentSimilarity}
          targetSimilarity={targetSimilarity}
        />
      </div>

      <div className="main-grid">
        {/* Input */}
        <div className="card">
          <h2>Input Message</h2>
          <textarea
            value={message}
            onChange={(e) => setMessage(e.target.value)}
            placeholder="Enter message to compress..."
          />
          
          <button className="btn-secondary" onClick={loadSample}>
            Load Sample
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
          
          <button 
            className="btn-primary"
            onClick={processMessage} 
            disabled={processing || !message.trim() || !mspEnabled}
          >
            {processing ? 'Processing...' : 'Run Pipeline'}
          </button>
          
          {error && <div className="error-message">{error}</div>}
        </div>

        {/* Refinement History */}
        <div className="card">
          <h2>Refinement Iterations</h2>
          {result ? (
            <div className="iterations">
              {result.refinement_history.map((step, i) => (
                <div key={i} className={`iteration ${step.similarity >= targetSimilarity ? 'passed' : ''}`}>
                  <div className="iter-header">
                    <span className="iter-num">Iteration {step.iteration}</span>
                    <span className="iter-intent">{step.intent}</span>
                    <span className={`iter-similarity ${step.similarity >= targetSimilarity ? 'good' : ''}`}>
                      {(step.similarity * 100).toFixed(1)}%
                    </span>
                  </div>
                  
                  <div className="iter-target">{step.target}</div>
                  
                  <div className="similarity-bar">
                    <div 
                      className="similarity-fill"
                      style={{ width: `${step.similarity * 100}%` }}
                    />
                    <div 
                      className="threshold-marker"
                      style={{ left: `${targetSimilarity * 100}%` }}
                    />
                  </div>
                  
                  {step.feedback && (
                    <div className="feedback-section">
                      <button 
                        className="btn-text"
                        onClick={() => setExpandedFeedback(expandedFeedback === i ? null : i)}
                      >
                        {expandedFeedback === i ? 'Hide' : 'Show'} Information Loss Analysis
                      </button>
                      {expandedFeedback === i && (
                        <div className="feedback-content">
                          {step.feedback}
                        </div>
                      )}
                    </div>
                  )}
                </div>
              ))}
              
              <div className={`convergence-status ${result.converged ? 'converged' : 'not-converged'}`}>
                {result.converged 
                  ? `Converged after ${result.iterations} iteration(s)` 
                  : `Did not converge (max ${result.iterations} iterations)`}
              </div>
            </div>
          ) : (
            <div className="placeholder">
              {processing ? 'Processing...' : 'Results will appear here'}
            </div>
          )}
        </div>

        {/* Output */}
        <div className="card">
          <h2>Compressed Signal</h2>
          {result ? (
            <>
              <div className="signal-output">
                <pre>{result.final_signal_json}</pre>
              </div>
              
              <h3>Agent B Response</h3>
              <div className="response-output">
                {result.agent_b_response}
              </div>
            </>
          ) : (
            <div className="placeholder">
              {processing ? 'Processing...' : 'Signal will appear here'}
            </div>
          )}
        </div>
      </div>

      {/* Metrics */}
      {result && (
        <div className="card metrics-card">
          <div className="metrics-row">
            <div className="metric">
              <span className="metric-value">{result.agent_a_tokens}</span>
              <span className="metric-label">Original Tokens</span>
            </div>
            <div className="metric">
              <span className="metric-value">{result.final_signal_tokens}</span>
              <span className="metric-label">Signal Tokens</span>
            </div>
            <div className="metric">
              <span className="metric-value">{result.tokens_saved}</span>
              <span className="metric-label">Tokens Saved</span>
            </div>
            <div className="metric">
              <span className="metric-value">{result.iterations}</span>
              <span className="metric-label">Iterations</span>
            </div>
            <div className="metric">
              <span className="metric-value">{(result.final_similarity * 100).toFixed(0)}%</span>
              <span className="metric-label">Final Similarity</span>
            </div>
            <div className="metric">
              <span className="metric-value">{(result.latency_ms / 1000).toFixed(1)}s</span>
              <span className="metric-label">Total Time</span>
            </div>
          </div>
        </div>
      )}

      {/* How It Works */}
      <div className="card how-it-works">
        <h2>How It Works</h2>
        <div className="how-grid">
          <div className="how-section">
            <h3>1. Encoding</h3>
            <p>LLM (Groq) extracts structured information from the message into a JSON signal with intent, target, params, and constraints.</p>
          </div>
          <div className="how-section">
            <h3>2. Decoding</h3>
            <p>LLM expands the JSON signal back into natural language text.</p>
          </div>
          <div className="how-section">
            <h3>3. Semantic Similarity</h3>
            <p>Uses <strong>sentence-transformers</strong> (all-MiniLM-L6-v2) to convert both texts into 384-dimensional vectors, then computes <strong>cosine similarity</strong>:</p>
            <div className="formula">
              similarity = (A · B) / (||A|| × ||B||)
            </div>
            <p>Result is 0-1 where 1 = identical meaning.</p>
          </div>
          <div className="how-section">
            <h3>4. Feedback Loop</h3>
            <p>If similarity &lt; threshold, LLM analyzes what information was lost and re-encodes with that feedback. Repeats until threshold met or max iterations.</p>
          </div>
        </div>
      </div>
    </div>
  )
}

export default App
