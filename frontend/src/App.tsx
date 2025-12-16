import { useState, useEffect, useRef } from 'react'
import './App.css'

interface Key {
  type: string
  value: string
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
        } else if (data.event === 'compression_start') {
          setCompletedStages(prev => new Set([...prev, 'input']))
          setActiveStage('compress')
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

  return (
    <div className="container">
      <header>
        <h1>🔬 Minimal Signaling Experiment</h1>
        <p className="subtitle">Study minimal information exchange between LLM agents</p>
      </header>

      {/* Pipeline Visualization */}
      <div className="card pipeline-card">
        <div className="pipeline">
          <div className={getStageClass('input')}>Agent A</div>
          <span className="arrow">→</span>
          <div className={getStageClass('compress')}>Compression</div>
          <span className="arrow">→</span>
          <div className={getStageClass('extract')}>Extraction</div>
          <span className="arrow">→</span>
          <div className={getStageClass('judge')}>Judge</div>
          <span className="arrow">→</span>
          <div className={getStageClass('output')}>Agent B</div>
        </div>
      </div>

      <div className="grid">
        {/* Input Panel */}
        <div className="card">
          <h2>📝 Input Message</h2>
          <textarea
            value={message}
            onChange={(e) => setMessage(e.target.value)}
            placeholder={`Enter a message to compress and extract semantic keys...

Example:
INSTRUCTION: Analyze the quarterly sales data
STATE: Data has been collected from all regions
GOAL: Identify trends and anomalies
CONTEXT: This is for the Q4 board meeting
CONSTRAINT: Report must be ready by Friday`}
          />
          
          <div className="config-row">
            <label>Token Budget:</label>
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
            {processing ? '⏳ Processing...' : '🚀 Process Message'}
          </button>
        </div>

        {/* Results Panel */}
        <div className="card">
          <h2>📊 Compression Results</h2>
          
          {result ? (
            <>
              <div className="compression-bar">
                <div 
                  className="compression-fill" 
                  style={{ width: `${result.compression_ratio * 100}%` }}
                />
                <span className="compression-label">
                  {(result.compression_ratio * 100).toFixed(1)}% of original
                </span>
              </div>
              
              <div className="metrics">
                <div className="metric">
                  <div className="metric-value">{result.original_tokens}</div>
                  <div className="metric-label">Original</div>
                </div>
                <div className="metric">
                  <div className="metric-value">{result.final_tokens}</div>
                  <div className="metric-label">Final</div>
                </div>
                <div className="metric">
                  <div className="metric-value">{result.passes}</div>
                  <div className="metric-label">Passes</div>
                </div>
                <div className="metric">
                  <div className="metric-value">{result.duration_ms.toFixed(1)}</div>
                  <div className="metric-label">ms</div>
                </div>
              </div>
              
              {result.compressed_text && (
                <div className="compressed-text">
                  <strong>Compressed:</strong> {result.compressed_text}
                </div>
              )}
            </>
          ) : (
            <p className="placeholder">Process a message to see results...</p>
          )}
        </div>

        {/* Semantic Keys Panel */}
        <div className="card">
          <h2>🔑 Extracted Semantic Keys</h2>
          <div className="keys-container">
            {result?.keys.length ? (
              result.keys.map((key, i) => (
                <div 
                  key={i} 
                  className="key"
                  style={{ borderLeftColor: keyColors[key.type] || '#666' }}
                >
                  <span className="key-type">{key.type}</span>
                  <span className="key-value">{key.value}</span>
                </div>
              ))
            ) : (
              <p className="placeholder">No keys extracted yet...</p>
            )}
          </div>
        </div>

        {/* Judge & Events Panel */}
        <div className="card">
          <h2>⚖️ Judge Verification</h2>
          {result && result.judge_passed !== null ? (
            <div className={`judge-result ${result.judge_passed ? 'passed' : 'failed'}`}>
              <span className="judge-icon">{result.judge_passed ? '✅' : '❌'}</span>
              <div>
                <strong>{result.judge_passed ? 'PASSED' : 'FAILED'}</strong>
                <br />
                <span className="confidence">
                  Confidence: {((result.judge_confidence ?? 0) * 100).toFixed(0)}%
                </span>
              </div>
            </div>
          ) : (
            <p className="placeholder">Judge result will appear here...</p>
          )}
          
          <h2 style={{ marginTop: '20px' }}>📡 Live Events</h2>
          <div className="events">
            {events.map((evt, i) => (
              <div key={i} className="event">
                <span className="event-name">{evt.event || evt.type}</span>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  )
}

export default App
