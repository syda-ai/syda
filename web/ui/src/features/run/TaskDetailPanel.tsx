import { useState } from 'react'
import type { GenerationTask, TaskAction, LogEntry } from './types'

type TaskDetailPanelProps = {
  task: GenerationTask
  onAction: (action: TaskAction) => void
  onClose: () => void
}

function getStatusIcon(status: string): string {
  switch (status) {
    case 'success': return '✅'
    case 'failed': return '❌'
    case 'running': return '🔄'
    case 'queued': return '⏳'
    case 'scheduled': return '📅'
    case 'skipped': return '⏭️'
    case 'up_for_retry': return '🔁'
    case 'upstream_failed': return '⬆️❌'
    default: return '⚪'
  }
}

function getStatusColor(status: string): string {
  switch (status) {
    case 'success': return 'var(--success)'
    case 'failed': return 'var(--danger)'
    case 'running': return 'var(--warn)'
    case 'queued': return 'var(--primary)'
    case 'scheduled': return 'var(--accent)'
    default: return 'var(--muted)'
  }
}

// Mock logs for demonstration
function generateMockLogs(task: GenerationTask): LogEntry[] {
  const logs: LogEntry[] = []
  const baseTime = task.execution?.startTime || new Date()
  
  logs.push({
    timestamp: new Date(baseTime.getTime()),
    level: 'INFO',
    message: `Starting ${task.schemaName} generation`,
    taskId: task.id
  })
  
  logs.push({
    timestamp: new Date(baseTime.getTime() + 1000),
    level: 'INFO',
    message: 'Loading schema configuration',
    taskId: task.id
  })
  
  logs.push({
    timestamp: new Date(baseTime.getTime() + 3000),
    level: 'INFO',
    message: `Initializing ${task.config.model}`,
    taskId: task.id
  })

  if (task.status === 'running') {
    logs.push({
      timestamp: new Date(baseTime.getTime() + 15000),
      level: 'INFO',
      message: `Generated batch 1/${Math.ceil(task.sampleSize / task.config.batchSize)} (${task.config.batchSize} records)`,
      taskId: task.id
    })
    
    logs.push({
      timestamp: new Date(baseTime.getTime() + 30000),
      level: 'INFO',
      message: `Generated batch 2/${Math.ceil(task.sampleSize / task.config.batchSize)} (${task.config.batchSize * 2} records)`,
      taskId: task.id
    })
  } else if (task.status === 'success') {
    const batches = Math.ceil(task.sampleSize / task.config.batchSize)
    for (let i = 1; i <= batches; i++) {
      logs.push({
        timestamp: new Date(baseTime.getTime() + i * 15000),
        level: 'INFO',
        message: `Generated batch ${i}/${batches} (${Math.min(i * task.config.batchSize, task.sampleSize)} records)`,
        taskId: task.id
      })
    }
    
    logs.push({
      timestamp: new Date(baseTime.getTime() + (batches + 1) * 15000),
      level: 'INFO',
      message: 'Validating referential integrity',
      taskId: task.id
    })
    
    logs.push({
      timestamp: new Date(baseTime.getTime() + (batches + 2) * 15000),
      level: 'INFO',
      message: `Task completed successfully (${task.sampleSize} records generated)`,
      taskId: task.id
    })
  } else if (task.status === 'failed') {
    logs.push({
      timestamp: new Date(baseTime.getTime() + 15000),
      level: 'ERROR',
      message: 'Generation failed: API rate limit exceeded',
      taskId: task.id
    })
  }

  return logs
}

export default function TaskDetailPanel({ task, onAction, onClose }: TaskDetailPanelProps) {
  const [activeTab, setActiveTab] = useState<'details' | 'logs' | 'config'>('details')
  const logs = generateMockLogs(task)

  const formatTimestamp = (date: Date) => {
    return date.toLocaleString('en-US', {
      year: 'numeric',
      month: '2-digit',
      day: '2-digit',
      hour: '2-digit',
      minute: '2-digit',
      second: '2-digit'
    })
  }

  const getLogLevelColor = (level: string) => {
    switch (level) {
      case 'ERROR': return 'var(--danger)'
      case 'WARN': return 'var(--warn)'
      case 'INFO': return 'var(--primary)'
      case 'DEBUG': return 'var(--muted)'
      default: return 'var(--text)'
    }
  }

  return (
    <div style={{
      position: 'fixed',
      top: 0,
      left: 0,
      right: 0,
      bottom: 0,
      background: 'rgba(0,0,0,0.5)',
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'center',
      zIndex: 1000
    }}>
      <div className="panel" style={{ 
        width: '90%', 
        maxWidth: 900, 
        height: '80%', 
        display: 'flex', 
        flexDirection: 'column',
        overflow: 'hidden'
      }}>
        {/* Header */}
        <div style={{ 
          padding: 24, 
          borderBottom: '1px solid var(--border)',
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center'
        }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 16 }}>
            <div>
              <h3 style={{ margin: 0, display: 'flex', alignItems: 'center', gap: 8 }}>
                Task: {task.schemaName}
              </h3>
              <div style={{ fontSize: '0.9rem', color: 'var(--muted)', marginTop: 4 }}>
                {task.sampleSize.toLocaleString()} rows • {task.config.model}
              </div>
            </div>
            
            <div style={{ 
              display: 'flex', 
              alignItems: 'center', 
              gap: 8,
              padding: '8px 12px',
              background: `${getStatusColor(task.status)}20`,
              border: `1px solid ${getStatusColor(task.status)}`,
              borderRadius: 8
            }}>
              <span>{getStatusIcon(task.status)}</span>
              <span style={{ fontWeight: 600, color: getStatusColor(task.status) }}>
                {task.status.toUpperCase()}
              </span>
              {task.execution?.duration && (
                <span style={{ fontSize: '0.8rem', color: 'var(--muted)' }}>
                  • {Math.floor(task.execution.duration / 60)}m {task.execution.duration % 60}s
                </span>
              )}
            </div>
          </div>

          <div style={{ display: 'flex', gap: 8 }}>
            <button 
              className="btn secondary"
              onClick={() => onAction('run')}
              disabled={task.status === 'running'}
            >
              ▶️ Run
            </button>
            <button 
              className="btn secondary"
              onClick={() => onAction('retry')}
              disabled={task.status !== 'failed'}
            >
              🔁 Retry
            </button>
            <button className="btn secondary" onClick={onClose}>
              ✕ Close
            </button>
          </div>
        </div>

        {/* Tabs */}
        <div style={{ 
          display: 'flex', 
          borderBottom: '1px solid var(--border)',
          background: 'var(--panel-2)'
        }}>
          {(['details', 'logs', 'config'] as const).map(tab => (
            <button
              key={tab}
              className={activeTab === tab ? 'btn' : 'btn secondary'}
              onClick={() => setActiveTab(tab)}
              style={{ 
                padding: '12px 20px', 
                fontSize: '0.9rem',
                textTransform: 'capitalize',
                borderRadius: 0,
                border: 'none',
                background: activeTab === tab ? 'var(--panel)' : 'transparent'
              }}
            >
              {tab === 'details' && '📊'} 
              {tab === 'logs' && '📋'} 
              {tab === 'config' && '⚙️'} 
              {tab}
            </button>
          ))}
        </div>

        {/* Content */}
        <div style={{ flex: 1, overflow: 'auto', padding: 24 }}>
          {activeTab === 'details' && (
            <div style={{ display: 'grid', gap: 24 }}>
              {/* Execution Details */}
              <div>
                <h4 style={{ margin: '0 0 12px 0' }}>🚀 Execution Details</h4>
                <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: 16 }}>
                  <div>
                    <div style={{ fontSize: '0.8rem', color: 'var(--muted)' }}>Start Time</div>
                    <div style={{ fontWeight: 600 }}>
                      {task.execution?.startTime ? formatTimestamp(task.execution.startTime) : 'Not started'}
                    </div>
                  </div>
                  <div>
                    <div style={{ fontSize: '0.8rem', color: 'var(--muted)' }}>End Time</div>
                    <div style={{ fontWeight: 600 }}>
                      {task.execution?.endTime ? formatTimestamp(task.execution.endTime) : 'Not finished'}
                    </div>
                  </div>
                  <div>
                    <div style={{ fontSize: '0.8rem', color: 'var(--muted)' }}>Records Generated</div>
                    <div style={{ fontWeight: 600 }}>
                      {task.execution?.recordsGenerated?.toLocaleString() || '0'}
                    </div>
                  </div>
                  <div>
                    <div style={{ fontSize: '0.8rem', color: 'var(--muted)' }}>Cost</div>
                    <div style={{ fontWeight: 600 }}>
                      ${task.execution?.cost?.toFixed(3) || '0.000'}
                    </div>
                  </div>
                </div>
              </div>

              {/* Dependencies */}
              <div>
                <h4 style={{ margin: '0 0 12px 0' }}>🔗 Dependencies</h4>
                {task.dependencies.length > 0 ? (
                  <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap' }}>
                    {task.dependencies.map(depId => (
                      <span key={depId} style={{
                        padding: '4px 8px',
                        background: 'var(--panel-2)',
                        border: '1px solid var(--border)',
                        borderRadius: 6,
                        fontSize: '0.85rem'
                      }}>
                        {depId.replace('task_', '')}
                      </span>
                    ))}
                  </div>
                ) : (
                  <div style={{ color: 'var(--muted)', fontSize: '0.9rem' }}>
                    No dependencies (root task)
                  </div>
                )}
              </div>
            </div>
          )}

          {activeTab === 'logs' && (
            <div>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 16 }}>
                <h4 style={{ margin: 0 }}>📋 Task Logs</h4>
                <div style={{ display: 'flex', gap: 8 }}>
                  <button className="btn secondary" style={{ padding: '4px 8px', fontSize: '0.8rem' }}>
                    📥 Download
                  </button>
                  <button className="btn secondary" style={{ padding: '4px 8px', fontSize: '0.8rem' }}>
                    🔄 Auto-refresh
                  </button>
                </div>
              </div>
              
              <div className="panel" style={{ 
                padding: 16, 
                background: '#000', 
                color: '#fff',
                fontFamily: 'ui-monospace, monospace',
                fontSize: '0.85rem',
                maxHeight: 400,
                overflow: 'auto'
              }}>
                {logs.map((log, index) => (
                  <div key={index} style={{ 
                    marginBottom: 4,
                    display: 'flex',
                    gap: 12
                  }}>
                    <span style={{ color: '#666', minWidth: 140 }}>
                      [{log.timestamp.toLocaleTimeString()}]
                    </span>
                    <span style={{ 
                      color: getLogLevelColor(log.level),
                      minWidth: 50,
                      fontWeight: 600
                    }}>
                      {log.level}:
                    </span>
                    <span>{log.message}</span>
                  </div>
                ))}
                
                {task.status === 'running' && (
                  <div style={{ 
                    marginTop: 8,
                    color: '#666',
                    display: 'flex',
                    alignItems: 'center',
                    gap: 8
                  }}>
                    <div style={{ 
                      width: 8, 
                      height: 8, 
                      background: '#f59e0b', 
                      borderRadius: '50%',
                      animation: 'pulse 1s infinite'
                    }} />
                    Generating data...
                  </div>
                )}
              </div>
            </div>
          )}

          {activeTab === 'config' && (
            <div style={{ display: 'grid', gap: 24 }}>
              {/* Generation Settings */}
              <div>
                <h4 style={{ margin: '0 0 12px 0' }}>🤖 Generation Settings</h4>
                <div className="panel" style={{ padding: 16 }}>
                  <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: 16 }}>
                    <div>
                      <div style={{ fontSize: '0.8rem', color: 'var(--muted)', marginBottom: 4 }}>AI Model</div>
                      <div style={{ fontWeight: 600 }}>{task.config.model}</div>
                    </div>
                    <div>
                      <div style={{ fontSize: '0.8rem', color: 'var(--muted)', marginBottom: 4 }}>Temperature</div>
                      <div style={{ fontWeight: 600 }}>{task.config.temperature}</div>
                    </div>
                    <div>
                      <div style={{ fontSize: '0.8rem', color: 'var(--muted)', marginBottom: 4 }}>Batch Size</div>
                      <div style={{ fontWeight: 600 }}>{task.config.batchSize}</div>
                    </div>
                    <div>
                      <div style={{ fontSize: '0.8rem', color: 'var(--muted)', marginBottom: 4 }}>Max Retries</div>
                      <div style={{ fontWeight: 600 }}>{task.config.maxRetries}</div>
                    </div>
                  </div>
                  
                  {task.config.seed && (
                    <div style={{ marginTop: 16 }}>
                      <div style={{ fontSize: '0.8rem', color: 'var(--muted)', marginBottom: 4 }}>Seed (Reproducible)</div>
                      <div style={{ fontWeight: 600, fontFamily: 'monospace' }}>{task.config.seed}</div>
                    </div>
                  )}
                </div>
              </div>

              {/* Custom Prompt */}
              {task.config.customPrompt && (
                <div>
                  <h4 style={{ margin: '0 0 12px 0' }}>🎯 Custom Prompt</h4>
                  <div className="panel" style={{ padding: 16 }}>
                    <div style={{ 
                      fontFamily: 'ui-monospace, monospace',
                      fontSize: '0.9rem',
                      lineHeight: 1.5,
                      background: 'var(--panel-2)',
                      padding: 12,
                      borderRadius: 8,
                      border: '1px solid var(--border)'
                    }}>
                      {task.config.customPrompt}
                    </div>
                  </div>
                </div>
              )}

              {/* Performance Metrics */}
              {task.execution && (
                <div>
                  <h4 style={{ margin: '0 0 12px 0' }}>📈 Performance Metrics</h4>
                  <div className="panel" style={{ padding: 16 }}>
                    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(150px, 1fr))', gap: 16 }}>
                      <div>
                        <div style={{ fontSize: '0.8rem', color: 'var(--muted)', marginBottom: 4 }}>Duration</div>
                        <div style={{ fontWeight: 600 }}>
                          {task.execution.duration ? 
                            `${Math.floor(task.execution.duration / 60)}m ${task.execution.duration % 60}s` : 
                            'N/A'
                          }
                        </div>
                      </div>
                      <div>
                        <div style={{ fontSize: '0.8rem', color: 'var(--muted)', marginBottom: 4 }}>Rate</div>
                        <div style={{ fontWeight: 600 }}>
                          {task.execution.duration ? 
                            `${(task.execution.recordsGenerated / task.execution.duration).toFixed(1)} rows/sec` : 
                            'N/A'
                          }
                        </div>
                      </div>
                      <div>
                        <div style={{ fontSize: '0.8rem', color: 'var(--muted)', marginBottom: 4 }}>Cost</div>
                        <div style={{ fontWeight: 600, color: 'var(--primary)' }}>
                          ${task.execution.cost.toFixed(3)}
                        </div>
                      </div>
                      <div>
                        <div style={{ fontSize: '0.8rem', color: 'var(--muted)', marginBottom: 4 }}>Try Number</div>
                        <div style={{ fontWeight: 600 }}>
                          {task.execution.tryNumber} / {task.config.maxRetries}
                        </div>
                      </div>
                    </div>
                  </div>
                </div>
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  )
}

function getLogLevelColor(level: string): string {
  switch (level) {
    case 'ERROR': return '#ef4444'
    case 'WARN': return '#f59e0b'
    case 'INFO': return '#3b82f6'
    case 'DEBUG': return '#6b7280'
    default: return '#ffffff'
  }
}
