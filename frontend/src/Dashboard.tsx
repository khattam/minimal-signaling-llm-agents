import React, { useState, useEffect } from 'react';
import './Dashboard.css';

const API_URL = import.meta.env.DEV ? 'http://localhost:8080' : '';

interface SectionBreakdown {
  title: string;
  importance: string;
  tokens: number;
  content_preview: string;
}

interface IterationHistory {
  iteration: number;
  similarity: number;
  tokens: number;
  compression: number;
  section_importances?: Array<{
    title: string;
    importance: string;
    key_concepts: string[];
  }>;
  feedback?: string;
}

interface DashboardData {
  metadata: {
    run_id?: string;
    timestamp: string;
    test_name: string;
    model: string;
  };
  success: boolean;
  iterations: number;
  original_tokens: number;
  final_tokens: number;
  compression_ratio: number;
  final_similarity: number;
  target_similarity: number;
  sections: {
    count: number;
    breakdown: SectionBreakdown[];
  };
  iteration_history: IterationHistory[];
  texts: {
    original: string;
    final_decoded: string;
    final_signal_json: string;
  };
}

interface RunMetadata {
  run_id: string;
  timestamp: string;
  success: boolean;
  iterations: number;
  final_similarity: number;
  compression_ratio: number;
  original_tokens: number;
  final_tokens: number;
}

const Dashboard: React.FC = () => {
  const [runs, setRuns] = useState<RunMetadata[]>([]);
  const [selectedRunId, setSelectedRunId] = useState<string | null>(null);
  const [data, setData] = useState<DashboardData | null>(null);
  const [activeTab, setActiveTab] = useState<'overview' | 'iterations' | 'sections' | 'comparison'>('overview');
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);

  // New run state
  const [showNewRun, setShowNewRun] = useState(false);
  const [newRunText, setNewRunText] = useState('');
  const [newRunSimilarity, setNewRunSimilarity] = useState(0.80);
  const [newRunIterations, setNewRunIterations] = useState(5);
  const [processing, setProcessing] = useState(false);

  useEffect(() => {
    loadRuns();
  }, []);

  useEffect(() => {
    if (selectedRunId) {
      loadRunData(selectedRunId);
    }
  }, [selectedRunId]);

  const loadRuns = async () => {
    try {
      setLoading(true);
      const res = await fetch(`${API_URL}/api/runs`);
      const runsList = await res.json();
      setRuns(runsList);
      
      if (runsList.length > 0 && !selectedRunId) {
        setSelectedRunId(runsList[0].run_id);
      }
    } catch (err) {
      setError('Failed to load runs: ' + (err as Error).message);
    } finally {
      setLoading(false);
    }
  };

  const loadRunData = async (runId: string) => {
    try {
      setLoading(true);
      const res = await fetch(`${API_URL}/api/runs/${runId}`);
      const runData = await res.json();
      setData(runData);
      setError(null);
    } catch (err) {
      setError('Failed to load run data: ' + (err as Error).message);
    } finally {
      setLoading(false);
    }
  };

  const deleteRun = async (runId: string) => {
    if (!confirm('Delete this run?')) return;
    
    try {
      await fetch(`${API_URL}/api/runs/${runId}`, { method: 'DELETE' });
      await loadRuns();
      
      if (selectedRunId === runId) {
        setSelectedRunId(runs[0]?.run_id || null);
      }
    } catch (err) {
      setError('Failed to delete run: ' + (err as Error).message);
    }
  };

  const runEncoding = async () => {
    if (!newRunText.trim()) return;
    
    setProcessing(true);
    setError(null);
    
    try {
      const res = await fetch(`${API_URL}/api/encode`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          text: newRunText,
          target_similarity: newRunSimilarity,
          max_iterations: newRunIterations
        })
      });
      
      if (!res.ok) throw new Error('Encoding failed');
      
      const result = await res.json();
      
      await loadRuns();
      setSelectedRunId(result.metadata.run_id);
      setShowNewRun(false);
      setNewRunText('');
    } catch (err) {
      setError('Failed to encode: ' + (err as Error).message);
    } finally {
      setProcessing(false);
    }
  };

  const getImportanceColor = (importance: string) => {
    switch (importance) {
      case 'critical': return '#ef4444';
      case 'high': return '#f97316';
      case 'medium': return '#eab308';
      case 'low': return '#22c55e';
      default: return '#6b7280';
    }
  };

  if (error && !data) {
    return <div className="error">⚠️ {error}</div>;
  }

  return (
    <div className="dashboard">
      <header className="dashboard-header">
        <h1>Hierarchical Adaptive Encoding</h1>
        <div className="metadata">
          <span>{runs.length} saved runs</span>
        </div>
      </header>

      <div className="dashboard-layout">
        {/* Sidebar */}
        <aside className="sidebar">
          <div className="sidebar-header">
            <h3>Runs</h3>
            <button className="btn-new-run" onClick={() => setShowNewRun(true)}>
              + New
            </button>
          </div>
          
          <div className="runs-list">
            {runs.map(run => (
              <div 
                key={run.run_id}
                className={`run-item ${selectedRunId === run.run_id ? 'active' : ''}`}
                onClick={() => setSelectedRunId(run.run_id)}
              >
                <div className="run-header">
                  <span className={`run-status ${run.success ? 'success' : 'failed'}`}>
                    {run.success ? '✓' : '✗'}
                  </span>
                  <span className="run-time">
                    {new Date(run.timestamp).toLocaleTimeString()}
                  </span>
                </div>
                <div className="run-stats">
                  <span>{run.iterations} iter</span>
                  <span>{(run.final_similarity * 100).toFixed(0)}%</span>
                </div>
                <button 
                  className="btn-delete-run"
                  onClick={(e) => {
                    e.stopPropagation();
                    deleteRun(run.run_id);
                  }}
                >
                  🗑️
                </button>
              </div>
            ))}
          </div>
        </aside>

        {/* Main content */}
        <main className="main-content">
          {showNewRun ? (
            <div className="new-run-panel">
              <div className="panel-header">
                <h2>New Encoding Run</h2>
                <button className="btn-close" onClick={() => setShowNewRun(false)}>✕</button>
              </div>
              
              <div className="form-group">
                <label>Input Text</label>
                <textarea
                  value={newRunText}
                  onChange={(e) => setNewRunText(e.target.value)}
                  placeholder="Enter text to encode..."
                  rows={12}
                />
              </div>
              
              <div className="form-row">
                <div className="form-group">
                  <label>Target Similarity: {(newRunSimilarity * 100).toFixed(0)}%</label>
                  <input
                    type="range"
                    min="0.6"
                    max="0.95"
                    step="0.05"
                    value={newRunSimilarity}
                    onChange={(e) => setNewRunSimilarity(parseFloat(e.target.value))}
                  />
                </div>
                
                <div className="form-group">
                  <label>Max Iterations: {newRunIterations}</label>
                  <input
                    type="range"
                    min="1"
                    max="10"
                    step="1"
                    value={newRunIterations}
                    onChange={(e) => setNewRunIterations(parseInt(e.target.value))}
                  />
                </div>
              </div>
              
              <button 
                className="btn-run"
                onClick={runEncoding}
                disabled={processing || !newRunText.trim()}
              >
                {processing ? 'Processing...' : 'Run Encoding'}
              </button>
              
              {error && <div className="error-message">{error}</div>}
            </div>
          ) : data ? (
            <>
              <div className="tabs">
                <button 
                  className={activeTab === 'overview' ? 'active' : ''} 
                  onClick={() => setActiveTab('overview')}
                >
                  Overview
                </button>
                <button 
                  className={activeTab === 'iterations' ? 'active' : ''} 
                  onClick={() => setActiveTab('iterations')}
                >
                  Iterations
                </button>
                <button 
                  className={activeTab === 'comparison' ? 'active' : ''} 
                  onClick={() => setActiveTab('comparison')}
                >
                  Comparison
                </button>
              </div>

              <div className="content">
                {activeTab === 'overview' && (
                  <div className="overview">
                    <div className="metrics-grid">
                      <div className="metric-card">
                        <div className="metric-value">{data.original_tokens}</div>
                        <div className="metric-label">Original Size</div>
                      </div>

                      <div className="metric-card">
                        <div className="metric-value">{data.final_tokens}</div>
                        <div className="metric-label">Compressed Size</div>
                      </div>

                      <div className="metric-card">
                        <div className="metric-value">{((1 - data.compression_ratio) * 100).toFixed(1)}%</div>
                        <div className="metric-label">Reduction</div>
                        <div className="metric-sub">{data.original_tokens - data.final_tokens} tokens saved</div>
                      </div>

                      <div className="metric-card">
                        <div className="metric-value">{(data.final_similarity * 100).toFixed(1)}%</div>
                        <div className="metric-label">Similarity</div>
                        <div className="metric-sub">Target: {(data.target_similarity * 100).toFixed(0)}%</div>
                      </div>

                      <div className="metric-card">
                        <div className="metric-value">{data.iterations}</div>
                        <div className="metric-label">Iterations</div>
                      </div>

                      <div className="metric-card">
                        <div className="metric-value">{data.sections.count}</div>
                        <div className="metric-label">Sections</div>
                      </div>
                    </div>

                    {/* Hierarchical Tree */}
                    {data.iteration_history[0]?.section_importances && (
                      <div className="tree-container">
                        <h3>Section Hierarchy</h3>
                        {data.iteration_history[0].section_importances.map((sec, idx) => (
                          <div key={idx} className="tree-node">
                            <div 
                              className="tree-node-header" 
                              style={{
                                borderLeftColor: 
                                  sec.importance === 'critical' ? '#dc3545' :
                                  sec.importance === 'high' ? '#fd7e14' :
                                  sec.importance === 'medium' ? '#ffc107' : '#198754'
                              }}
                            >
                              <span className="tree-node-title">{sec.title}</span>
                              <span className={`tree-node-badge ${sec.importance}`}>
                                {sec.importance}
                              </span>
                            </div>
                            <div className="tree-node-concepts">
                              {sec.key_concepts.map((concept, cidx) => (
                                <span key={cidx} className="concept-tag">{concept}</span>
                              ))}
                            </div>
                          </div>
                        ))}
                      </div>
                    )}
                  </div>
                )}

                {activeTab === 'iterations' && (
                  <div className="iterations">
                    <h3>Iteration Details</h3>
                    <div className="iteration-chart">
                      {data.iteration_history.map((iter, idx) => (
                        <div key={idx} className="iteration-row">
                          <div className="iteration-number">Iteration {iter.iteration}</div>
                          <div className="iteration-bars">
                            <div className="bar-group">
                              <div className="bar-label">Kept: {(iter.compression * 100).toFixed(1)}% ({iter.tokens} tokens)</div>
                              <div className="bar-container">
                                <div 
                                  className="bar" 
                                  style={{width: `${iter.compression * 100}%`, background: '#0d6efd'}}
                                />
                              </div>
                            </div>
                            <div className="bar-group">
                              <div className="bar-label">Reduced: {((1 - iter.compression) * 100).toFixed(1)}%</div>
                              <div className="bar-container">
                                <div 
                                  className="bar" 
                                  style={{width: `${(1 - iter.compression) * 100}%`, background: '#6c757d'}}
                                />
                              </div>
                            </div>
                            <div className="bar-group">
                              <div className="bar-label">Similarity: {(iter.similarity * 100).toFixed(1)}%</div>
                              <div className="bar-container">
                                <div 
                                  className="bar" 
                                  style={{
                                    width: `${iter.similarity * 100}%`,
                                    background: iter.similarity >= data.target_similarity ? '#198754' : '#fd7e14'
                                  }}
                                />
                              </div>
                            </div>
                          </div>
                          {iter.feedback && (
                            <div className="iteration-feedback">
                              <strong>Feedback:</strong>
                              <pre>{iter.feedback}</pre>
                            </div>
                          )}
                        </div>
                      ))}
                    </div>
                  </div>
                )}

                {activeTab === 'comparison' && (
                  <div className="comparison">
                    <div className="comparison-grid">
                      <div className="comparison-panel">
                        <h3>Original Text ({data.original_tokens} tokens)</h3>
                        <pre className="text-content">{data.texts.original}</pre>
                      </div>
                      <div className="comparison-panel">
                        <h3>Decoded Text</h3>
                        <pre className="text-content">{data.texts.final_decoded}</pre>
                      </div>
                    </div>
                    <div className="signal-panel">
                      <h3>Compressed Signal (JSON)</h3>
                      <pre className="signal-content">{data.texts.final_signal_json}</pre>
                    </div>
                  </div>
                )}
              </div>
            </>
          ) : (
            <div className="placeholder">
              {loading ? 'Loading...' : 'Select a run or create a new one'}
            </div>
          )}
        </main>
      </div>
    </div>
  );
};

export default Dashboard;
