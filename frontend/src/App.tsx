import { useState, useEffect, useRef } from 'react'
import './App.css'

interface Key {
  type: string
  value: string
}

interface CompressionPass {
  pass: number
  tokens: number
  text: string
}

interface ProcessResult {
  success: boolean
  original_tokens: number
  final_tokens: number
  compression_ratio: number
  passes: number
  keys_extracted: number
  keys: Key[]
  compressed_text: string | null
  judge_passed: boolean | null
  judge_confidence: number | null
  duration_ms: number
}

interface Event {
  type: string
  event?: string
  timestamp: string
  data?: Record<string, unknown>
}

const API_URL = import.meta.env.DEV ? 'http://localhost:8080' : ''
const WS_URL = import.meta.env.DEV ? 'ws://localhost:8080/ws' : `ws://${window.location.host}/ws`

function App() {
  const [message, setMessage] = useState('')
  const [budget, setBudget] = useState(50)
  const [result, setResult] = useState<ProcessResult | null>(null)
  const [events, setEvents] = useState<Event[]>([])
  const [processing, setProcessing] = useState(false)
  const [activeStage, setActiveStage] = useState<string | null>(null)
  const [completedStages, setCompletedStages] = useState<Set<string>>(new Set())
  const [compressionPasses, setCompressionPasses] = useState<CompressionPass[]>([])
  const wsRef = useRef<WebSocket | null>(null)

  useEffect(() => {
    const connect = () => {
      const ws = new WebSocket(WS_URL)
      wsRef.current = ws
      
      ws.onmessage = (e) => {
        const data: Event = JSON.parse(e.data)
        setEvents(prev => [data, ...prev].slice(0, 50))
        
        // Update pipeline visualization
        if (data.event === 'message_received') {
          setActiveStage('input')
          setCompletedStages(new Set())
          setCompressionPasses([])
        } else if (data.event === 'compression_start') {
          setCompletedStages(prev => new Set([...prev, 'input']))
          setActiveStage('compress')
        } else if (data.event === 'compression_pass') {
          const passData = data.data as { pass_number: number; output_tokens: number; text?: string }
          setCompressionPasses(prev => [...prev, {
            pass: passData.pass_number,
            tokens: passData.output_tokens,
            text: passData.text || ''
          }])
        } else if (data.event === 'compression_complete') {
          setCompletedStages(prev => new Set([...prev, 'compress']))
        } else if (data.event === 'extraction_start') {
          setActiveStage('extract')
        } else if (data.event === 'extraction_complete') {
          setCompletedStages(prev => new Set([...prev, 'extract']))
        } else if (data.event === 'judge_start') {
          setActiveStage('judge')
        } else if (data.event === 'judge_complete') {
          setCompletedStages(prev => new Set([...prev, 'judge']))
        } else if (data.event === 'pipeline_complete') {
          setActiveStage(null)
          setCompletedStages(prev => new Set([...prev, 'output']))
        }
      }
      
      ws.onclose = () => setTimeout(connect, 1000)
    }
    connect()
    return () => wsRef.current?.close()
  }, [])

  const processMessage = async () => {
    if (!message.trim()) return
    
    setProcessing(true)
    setResult(null)
    setEvents([])
    setActiveStage(null)
    setCompletedStages(new Set())
    setCompressionPasses([])
    
    try {
      const res = await fetch(`${API_URL}/api/process`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message, budget })
      })
      const data = await res.json()
      setResult(data)
    } catch (err) {
      console.error(err)
    } finally {
      setProcessing(false)
    }
  }

  const getStageClass = (stage: string) => {
    if (activeStage === stage) return 'stage active'
    if (completedStages.has(stage)) return 'stage complete'
    return 'stage'
  }

  const keyColors: Record<string, string> = {
    INSTRUCTION: '#f59e0b',
    STATE: '#22c55e',
    GOAL: '#38bdf8',
    CONTEXT: '#a855f7',
    CONSTRAINT: '#ef4444'
  }

  const maxTokens = result?.original_tokens || compressionPasses[0]?.tokens || 100

  return (
    <div className="container">
      <header>
        <h1>🔬 Minimal Signaling Lab</h1>
        <p className="subtitle">Real-time visualization of LLM agent communication compression</p>
      </header>

      {/* Pipeline Visualization */}
      <div className="card pipeline-card">
        <div className="pipeline">
          <div className={getStageClass('input')}>
            <div className="stage-label">Source</div>
            <div className="stage-name">Agent A</div>
          </div>
          <span className="arrow">→</span>
          <div className={getStageClass('compress')}>
            <div className="stage-label">DistilBART</div>
            <div className="stage-name">Compress</div>
          </div>
          <span className="arrow">→</span>
          <div className={getStageClass('extract')}>
            <div className="stage-label">Semantic</div>
            <div className="stage-name">Extract</div>
          </div>
          <span className="arrow">→</span>
          <div className={getStageClass('judge')}>
            <div className="stage-label">Verify</div>
            <div className="stage-name">Judge</div>
          </div>
          <span className="arrow">→</span>
          <div className={getStageClass('output')}>
            <div className="stage-label">Target</div>
            <div className="stage-name">Agent B</div>
          </div>
        </div>
      </div>

      <div className="main-layout">
        {/* Left Panel - Input */}
        <div className="left-panel">
          <div className="card">
            <h2>📝 Agent A Message</h2>
            <textarea
              value={message}
              onChange={(e) => setMessage(e.target.value)}
              placeholder={`Enter a structured message to compress...

Example:
INSTRUCTION: Analyze the quarterly sales data
STATE: Data collected from all regions
GOAL: Identify trends and anomalies
CONTEXT: Q4 board meeting preparation
CONSTRAINT: Report due by Friday`}
            />
            
            <div className="config-row">
              <label>Token Budget</label>
              <input
                type="range"
                min="10"
                max="200"
                value={budget}
                onChange={(e) => setBudget(Number(e.target.value))}
              />
              <span className="value">{budget}</span>
            </div>
            
            <button onClick={processMessage} disabled={processing || !message.trim()}>
              {processing ? '⏳ Processing...' : '🚀 Compress & Extract'}
            </button>
          </div>

          {/* Events Log */}
          <div className="card">
            <h2>📡 Event Stream</h2>
            <div className="events-log">
              {events.length > 0 ? events.map((evt, i) => (
                <div key={i} className="event">
                  <span className="event-dot" />
                  <span className="event-name">{evt.event || evt.type}</span>
                </div>
              )) : (
                <p className="placeholder">Events will appear here...</p>
              )}
            </div>
          </div>
        </div>

        {/* Right Panel - Visualization */}
        <div className="right-panel">
          {/* Compression Passes */}
          <div className="card compression-viz">
            <h2>🔄 Compression Passes</h2>
            {compressionPasses.length > 0 || result ? (
              <div className="passes-container">
                {/* Original */}
                <div className="pass-item">
                  <span className="pass-label">Original</span>
                  <div className="pass-bar-container">
                    <div className="pass-bar" style={{ width: '100%' }} />
                  </div>
                  <span className="pass-tokens">{result?.original_tokens || '—'} tokens</span>
                </div>
                
                {/* Each compression pass */}
                {compressionPasses.map((pass, i) => (
                  <div key={i} className="pass-item">
                    <span className="pass-label">Pass {pass.pass}</span>
                    <div className="pass-bar-container">
                      <div 
                        className="pass-bar" 
                        style={{ width: `${(pass.tokens / maxTokens) * 100}%` }} 
                      />
                    </div>
                    <span className="pass-tokens">{pass.tokens} tokens</span>
                  </div>
                ))}
                
                {/* Final result */}
                {result && (
                  <div className="pass-item" style={{ borderColor: '#22c55e' }}>
                    <span className="pass-label" style={{ color: '#22c55e' }}>Final</span>
                    <div className="pass-bar-container">
                      <div 
                        className="pass-bar" 
                        style={{ 
                          width: `${result.compression_ratio * 100}%`,
                          background: 'linear-gradient(90deg, #22c55e, #4ade80)'
                        }} 
                      />
                    </div>
                    <span className="pass-tokens">{result.final_tokens} tokens</span>
                  </div>
                )}
              </div>
            ) : (
              <p className="placeholder">Compression visualization will appear here...</p>
            )}
          </div>

          {/* Results Metrics */}
          {result && (
            <div className="card">
              <h2>📊 Results</h2>
              <div className="results-grid">
                <div className="metric">
                  <div className="metric-value">{result.original_tokens}</div>
                  <div className="metric-label">Original</div>
                </div>
                <div className="metric">
                  <div className="metric-value highlight">{result.final_tokens}</div>
                  <div className="metric-label">Compressed</div>
                </div>
                <div className="metric">
                  <div className="metric-value success">{(result.compression_ratio * 100).toFixed(0)}%</div>
                  <div className="metric-label">Ratio</div>
                </div>
                <div className="metric">
                  <div className="metric-value">{result.duration_ms.toFixed(0)}</div>
                  <div className="metric-label">ms</div>
                </div>
              </div>
              
              {result.compressed_text && (
                <div className="compressed-output">
                  <strong style={{ color: '#38bdf8' }}>Compressed Output:</strong>
                  <br /><br />
                  {result.compressed_text}
                </div>
              )}
            </div>
          )}

          {/* Bottom Grid - Keys & Judge */}
          <div className="bottom-grid">
            {/* Semantic Keys */}
            <div className="card">
              <h2>🔑 Semantic Keys</h2>
              <div className="keys-grid">
                {result?.keys.length ? (
                  result.keys.map((key, i) => (
                    <div 
                      key={i} 
                      className="key"
                      style={{ borderLeftColor: keyColors[key.type] || '#666' }}
                    >
                      <span 
                        className="key-type"
                        style={{ color: keyColors[key.type] || '#666' }}
                      >
                        {key.type}
                      </span>
                      <span className="key-value">{key.value}</span>
                    </div>
                  ))
                ) : (
                  <p className="placeholder">Keys will appear here...</p>
                )}
              </div>
            </div>

            {/* Judge */}
            <div className="card">
              <h2>⚖️ Judge Verification</h2>
              {result && result.judge_passed !== null ? (
                <div className={`judge-result ${result.judge_passed ? 'passed' : 'failed'}`}>
                  <span className="judge-icon">{result.judge_passed ? '✅' : '❌'}</span>
                  <div className="judge-details">
                    <h3>{result.judge_passed ? 'Semantics Preserved' : 'Information Lost'}</h3>
                    <span style={{ color: '#94a3b8' }}>
                      Confidence: {((result.judge_confidence ?? 0) * 100).toFixed(0)}%
                    </span>
                    <div className="confidence-bar">
                      <div 
                        className="confidence-fill" 
                        style={{ width: `${(result.judge_confidence ?? 0) * 100}%` }}
                      />
                    </div>
                  </div>
                </div>
              ) : (
                <p className="placeholder">Judge result will appear here...</p>
              )}
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}

export default App
