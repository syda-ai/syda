import { useMemo, useState, useCallback } from 'react'
import ReactFlow, { Background, Controls, MiniMap, Position } from 'reactflow'
import 'reactflow/dist/style.css'
import dagre from 'dagre'
import type { SchemaProject, GenerationTask, TaskStatus, DAGViewMode, TaskAction } from './types'

// @ts-ignore - dagre doesn't have types
const dagreGraph = dagre

type DAGViewProps = {
  project: SchemaProject
  onTaskAction: (taskId: string, action: TaskAction) => void
  onProjectAction: (action: string) => void
}

const nodeWidth = 200
const nodeHeight = 80

// Layout tasks using Dagre
function layoutTasks(tasks: GenerationTask[]): { nodes: any[]; edges: any[] } {
  const g = new dagreGraph.graphlib.Graph()
  g.setDefaultEdgeLabel(() => ({}))
  g.setGraph({ rankdir: 'LR', nodesep: 50, ranksep: 100 })

  // Add nodes
  tasks.forEach(task => {
    g.setNode(task.id, { width: nodeWidth, height: nodeHeight })
  })

  // Add edges based on dependencies
  tasks.forEach(task => {
    task.dependencies.forEach(depId => {
      g.setEdge(depId, task.id)
    })
  })

  dagreGraph.layout(g)

  // Create React Flow nodes
  const nodes: any[] = tasks.map(task => {
    const nodePosition = g.node(task.id)
    return {
      id: task.id,
      type: 'custom',
      position: { 
        x: nodePosition.x - nodeWidth / 2, 
        y: nodePosition.y - nodeHeight / 2 
      },
      data: { 
        task,
        onAction: (action: TaskAction) => onTaskAction(task.id, action)
      },
      sourcePosition: Position.Right,
      targetPosition: Position.Left
    }
  })

  // Create React Flow edges
  const edges: any[] = []
  tasks.forEach(task => {
    task.dependencies.forEach(depId => {
      edges.push({
        id: `${depId}-${task.id}`,
        source: depId,
        target: task.id,
        type: 'smoothstep',
        animated: task.status === 'running',
        style: { 
          stroke: getEdgeColor(task.status),
          strokeWidth: 2
        },
        markerEnd: { type: 'arrowclosed' }
      })
    })
  })

  return { nodes, edges }
}

function getEdgeColor(status: TaskStatus): string {
  switch (status) {
    case 'success': return '#22c55e'
    case 'failed': return '#ef4444'
    case 'running': return '#f59e0b'
    case 'queued': return '#3b82f6'
    default: return '#6b7280'
  }
}

function getStatusIcon(status: TaskStatus): string {
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

function getStatusColor(status: TaskStatus): string {
  switch (status) {
    case 'success': return '#22c55e'
    case 'failed': return '#ef4444'
    case 'running': return '#f59e0b'
    case 'queued': return '#3b82f6'
    case 'scheduled': return '#8b5cf6'
    case 'skipped': return '#6b7280'
    case 'up_for_retry': return '#f97316'
    case 'upstream_failed': return '#dc2626'
    default: return '#9ca3af'
  }
}

// Custom Task Node Component
function TaskNode({ data }: { data: { task: GenerationTask; onAction: (action: TaskAction) => void } }) {
  const { task, onAction } = data
  const [showActions, setShowActions] = useState(false)

  const statusColor = getStatusColor(task.status)
  const isRunning = task.status === 'running'

  return (
    <div
      style={{
        width: nodeWidth,
        height: nodeHeight,
        background: 'var(--panel)',
        border: `2px solid ${statusColor}`,
        borderRadius: 12,
        padding: 12,
        boxShadow: isRunning ? `0 0 20px ${statusColor}40` : 'var(--shadow)',
        position: 'relative',
        cursor: 'pointer',
        transition: 'all 0.3s ease'
      }}
      onMouseEnter={() => setShowActions(true)}
      onMouseLeave={() => setShowActions(false)}
    >
      {/* Status indicator */}
      <div style={{
        position: 'absolute',
        top: -8,
        right: -8,
        width: 24,
        height: 24,
        background: statusColor,
        borderRadius: '50%',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        fontSize: '0.8rem',
        animation: isRunning ? 'pulse 2s infinite' : 'none'
      }}>
        {getStatusIcon(task.status)}
      </div>

      {/* Task info */}
      <div style={{ marginBottom: 8 }}>
        <div style={{ fontWeight: 700, fontSize: '0.95rem', marginBottom: 4 }}>
          {task.schemaName}
        </div>
        <div style={{ fontSize: '0.8rem', color: 'var(--muted)' }}>
          {task.sampleSize.toLocaleString()} rows
        </div>
      </div>

      {/* Progress bar for running tasks */}
      {isRunning && (
        <div style={{ marginBottom: 8 }}>
          <div style={{ 
            width: '100%', 
            height: 4, 
            background: 'var(--border)', 
            borderRadius: 2,
            overflow: 'hidden'
          }}>
            <div style={{ 
              width: '60%', // Mock progress
              height: '100%', 
              background: statusColor,
              animation: 'progress 2s ease-in-out infinite'
            }} />
          </div>
          <div style={{ fontSize: '0.75rem', color: 'var(--muted)', marginTop: 2 }}>
            60% • ETA: 45s
          </div>
        </div>
      )}

      {/* Duration/cost info */}
      <div style={{ fontSize: '0.75rem', color: 'var(--muted)' }}>
        {task.execution?.duration ? (
          <>Duration: {Math.floor(task.execution.duration / 60)}m {task.execution.duration % 60}s</>
        ) : (
          <>Model: {task.config.model.split(' ')[0]}</>
        )}
      </div>

      {/* Action buttons on hover */}
      {showActions && (
        <div style={{
          position: 'absolute',
          bottom: -40,
          left: 0,
          right: 0,
          display: 'flex',
          gap: 4,
          justifyContent: 'center',
          background: 'var(--panel)',
          padding: '4px 8px',
          borderRadius: 8,
          border: '1px solid var(--border)',
          boxShadow: 'var(--shadow)'
        }}>
          <button
            className="btn secondary"
            onClick={() => onAction('run')}
            style={{ padding: '2px 6px', fontSize: '0.7rem' }}
            title="Run task"
          >
            ▶️
          </button>
          <button
            className="btn secondary"
            onClick={() => onAction('logs')}
            style={{ padding: '2px 6px', fontSize: '0.7rem' }}
            title="View logs"
          >
            📋
          </button>
          <button
            className="btn secondary"
            onClick={() => onAction('edit')}
            style={{ padding: '2px 6px', fontSize: '0.7rem' }}
            title="Edit task"
          >
            ⚙️
          </button>
        </div>
      )}
    </div>
  )
}

// Register custom node type
const nodeTypes = {
  custom: TaskNode
}

export default function DAGView({ project, onTaskAction, onProjectAction }: DAGViewProps) {
  const [viewMode, setViewMode] = useState<DAGViewMode>('graph')
  const [selectedTask, setSelectedTask] = useState<string | null>(null)

  const { nodes, edges } = useMemo(() => layoutTasks(project.tasks), [project.tasks])

  const handleNodeClick = useCallback((event: any, node: any) => {
    setSelectedTask(node.id)
  }, [])

  const getProjectStatus = (): TaskStatus => {
    const taskStatuses = project.tasks.map(t => t.status)
    if (taskStatuses.some(s => s === 'failed')) return 'failed'
    if (taskStatuses.some(s => s === 'running')) return 'running'
    if (taskStatuses.every(s => s === 'success')) return 'success'
    if (taskStatuses.some(s => s === 'queued')) return 'queued'
    return 'none'
  }

  const projectStatus = getProjectStatus()
  const runningTasks = project.tasks.filter(t => t.status === 'running').length
  const completedTasks = project.tasks.filter(t => t.status === 'success').length

  return (
    <div style={{ height: '100%', display: 'flex', flexDirection: 'column' }}>
      {/* DAG Header */}
      <div style={{ 
        padding: '16px 20px', 
        borderBottom: '1px solid var(--border)',
        background: 'var(--panel)',
        display: 'flex',
        justifyContent: 'space-between',
        alignItems: 'center'
      }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 16 }}>
          <div>
            <h2 style={{ margin: 0, display: 'flex', alignItems: 'center', gap: 8 }}>
              {project.name}
            </h2>
            <div style={{ fontSize: '0.9rem', color: 'var(--muted)', marginTop: 4 }}>
              {project.tasks.length} tasks • Model: {project.defaultSettings.model} • Last run: {project.lastRun ? formatRelativeTime(project.lastRun.startTime) : 'Never'}
            </div>
          </div>
          
          <div style={{ 
            display: 'flex', 
            alignItems: 'center', 
            gap: 8,
            padding: '8px 12px',
            background: `${getStatusColor(projectStatus)}20`,
            border: `1px solid ${getStatusColor(projectStatus)}`,
            borderRadius: 8
          }}>
            <span>{getStatusIcon(projectStatus)}</span>
            <span style={{ fontWeight: 600, color: getStatusColor(projectStatus) }}>
              {projectStatus.toUpperCase()}
            </span>
            {runningTasks > 0 && (
              <span style={{ fontSize: '0.8rem', color: 'var(--muted)' }}>
                ({completedTasks}/{project.tasks.length} complete)
              </span>
            )}
          </div>
        </div>

        <div style={{ display: 'flex', gap: 8 }}>
          <button 
            className="btn secondary"
            onClick={() => onProjectAction('model')}
            style={{ display: 'flex', alignItems: 'center', gap: 6 }}
          >
            🤖 Model Config
          </button>
          <button 
            className="btn"
            onClick={() => onProjectAction('trigger')}
            disabled={projectStatus === 'running'}
            style={{ display: 'flex', alignItems: 'center', gap: 6 }}
          >
            ▶️ Trigger DAG
          </button>
          <button 
            className="btn secondary"
            onClick={() => onProjectAction('pause')}
            style={{ display: 'flex', alignItems: 'center', gap: 6 }}
          >
            ⏸️ Pause
          </button>
          <button 
            className="btn secondary"
            onClick={() => onProjectAction('refresh')}
            style={{ display: 'flex', alignItems: 'center', gap: 6 }}
          >
            🔄 Refresh
          </button>
        </div>
      </div>

      {/* View Mode Tabs */}
      <div style={{ 
        padding: '12px 20px', 
        borderBottom: '1px solid var(--border)',
        background: 'var(--panel-2)',
        display: 'flex',
        gap: 8
      }}>
        {(['graph', 'tree', 'gantt', 'details'] as DAGViewMode[]).map(mode => (
          <button
            key={mode}
            className={viewMode === mode ? 'btn' : 'btn secondary'}
            onClick={() => setViewMode(mode)}
            style={{ 
              padding: '6px 12px', 
              fontSize: '0.85rem',
              textTransform: 'capitalize',
              display: 'flex',
              alignItems: 'center',
              gap: 6
            }}
          >
            {mode === 'graph' && '📊'}
            {mode === 'tree' && '🌳'}
            {mode === 'gantt' && '📈'}
            {mode === 'details' && '📋'}
            {mode}
          </button>
        ))}
      </div>

      {/* Main Content */}
      <div style={{ flex: 1, position: 'relative' }}>
        {viewMode === 'graph' && (
          <ReactFlow
            nodes={nodes}
            edges={edges}
            onNodeClick={handleNodeClick}
            nodeTypes={nodeTypes}
            fitView
            style={{ background: 'transparent' }}
          >
            <Background variant={'dots' as any} gap={16} size={1} color="var(--border)" />
            <Controls />
            <MiniMap 
              nodeStrokeColor="var(--border)"
              nodeColor="var(--panel)"
              maskColor="rgba(0,0,0,0.1)"
            />
          </ReactFlow>
        )}

        {viewMode === 'tree' && (
          <div style={{ padding: 20, overflow: 'auto', height: '100%' }}>
            <div style={{ fontFamily: 'ui-monospace, monospace', fontSize: '0.9rem' }}>
              <div style={{ fontWeight: 700, marginBottom: 16, display: 'flex', alignItems: 'center', gap: 8 }}>
                📊 {project.name} Pipeline
              </div>
              {project.tasks.map((task, index) => (
                <div key={task.id} style={{ 
                  display: 'flex', 
                  alignItems: 'center', 
                  gap: 12, 
                  padding: '8px 0',
                  borderLeft: index < project.tasks.length - 1 ? '2px solid var(--border)' : 'none',
                  paddingLeft: 16,
                  marginLeft: 8
                }}>
                  <span>{index < project.tasks.length - 1 ? '├─' : '└─'}</span>
                  <span>{getStatusIcon(task.status)}</span>
                  <span style={{ fontWeight: 600 }}>{task.schemaName}</span>
                  <span style={{ color: 'var(--muted)' }}>({task.sampleSize.toLocaleString()} rows)</span>
                  <span style={{ 
                    color: getStatusColor(task.status),
                    fontWeight: 600,
                    textTransform: 'capitalize'
                  }}>
                    {task.status}
                  </span>
                  {task.execution?.duration && (
                    <span style={{ color: 'var(--muted)' }}>
                      - {Math.floor(task.execution.duration / 60)}m {task.execution.duration % 60}s
                    </span>
                  )}
                </div>
              ))}
            </div>
          </div>
        )}

        {viewMode === 'gantt' && (
          <div style={{ padding: 20, overflow: 'auto', height: '100%' }}>
            <h3 style={{ margin: '0 0 16px 0' }}>📈 Task Timeline</h3>
            <div className="panel" style={{ padding: 16 }}>
              {project.tasks.map(task => (
                <div key={task.id} style={{ 
                  display: 'grid', 
                  gridTemplateColumns: '200px 1fr 100px', 
                  gap: 16, 
                  alignItems: 'center',
                  padding: '12px 0',
                  borderBottom: '1px solid var(--border)'
                }}>
                  <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                    <span>{getStatusIcon(task.status)}</span>
                    <span style={{ fontWeight: 600 }}>{task.schemaName}</span>
                  </div>
                  
                  <div style={{ position: 'relative', height: 20 }}>
                    <div style={{ 
                      width: '100%', 
                      height: 6, 
                      background: 'var(--border)', 
                      borderRadius: 3,
                      position: 'absolute',
                      top: 7
                    }} />
                    <div style={{ 
                      width: task.status === 'success' ? '100%' : 
                             task.status === 'running' ? '60%' : '0%',
                      height: 6, 
                      background: getStatusColor(task.status), 
                      borderRadius: 3,
                      position: 'absolute',
                      top: 7,
                      transition: 'width 0.5s ease'
                    }} />
                  </div>
                  
                  <div style={{ fontSize: '0.85rem', color: 'var(--muted)', textAlign: 'right' }}>
                    {task.execution?.duration ? 
                      `${Math.floor(task.execution.duration / 60)}m ${task.execution.duration % 60}s` : 
                      'Not started'
                    }
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}

        {viewMode === 'details' && (
          <div style={{ padding: 20, overflow: 'auto', height: '100%' }}>
            <div style={{ display: 'grid', gap: 24 }}>
              {/* Project Summary */}
              <div className="panel" style={{ padding: 20 }}>
                <h3 style={{ margin: '0 0 16px 0' }}>📊 Project Summary</h3>
                <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: 16 }}>
                  <div>
                    <div style={{ fontSize: '0.8rem', color: 'var(--muted)', marginBottom: 4 }}>Total Records</div>
                    <div style={{ fontSize: '1.5rem', fontWeight: 700 }}>
                      {project.stats.totalRecords.toLocaleString()}
                    </div>
                  </div>
                  <div>
                    <div style={{ fontSize: '0.8rem', color: 'var(--muted)', marginBottom: 4 }}>Success Rate</div>
                    <div style={{ fontSize: '1.5rem', fontWeight: 700, color: 'var(--success)' }}>
                      {Math.round(project.stats.successRate * 100)}%
                    </div>
                  </div>
                  <div>
                    <div style={{ fontSize: '0.8rem', color: 'var(--muted)', marginBottom: 4 }}>Avg Duration</div>
                    <div style={{ fontSize: '1.5rem', fontWeight: 700 }}>
                      {Math.floor(project.stats.avgDuration / 60)}m {Math.floor(project.stats.avgDuration % 60)}s
                    </div>
                  </div>
                  <div>
                    <div style={{ fontSize: '0.8rem', color: 'var(--muted)', marginBottom: 4 }}>Total Cost</div>
                    <div style={{ fontSize: '1.5rem', fontWeight: 700, color: 'var(--primary)' }}>
                      ${project.stats.totalCost.toFixed(2)}
                    </div>
                  </div>
                </div>
              </div>

              {/* Task Performance */}
              <div className="panel" style={{ padding: 20 }}>
                <h3 style={{ margin: '0 0 16px 0' }}>⚡ Task Performance</h3>
                <table style={{ width: '100%', borderCollapse: 'collapse' }}>
                  <thead>
                    <tr style={{ borderBottom: '2px solid var(--border)' }}>
                      <th style={{ padding: '12px', textAlign: 'left' }}>Task</th>
                      <th style={{ padding: '12px', textAlign: 'center' }}>Status</th>
                      <th style={{ padding: '12px', textAlign: 'center' }}>Duration</th>
                      <th style={{ padding: '12px', textAlign: 'center' }}>Records</th>
                      <th style={{ padding: '12px', textAlign: 'center' }}>Cost</th>
                    </tr>
                  </thead>
                  <tbody>
                    {project.tasks.map(task => (
                      <tr key={task.id} style={{ borderBottom: '1px solid var(--border)' }}>
                        <td style={{ padding: '12px' }}>
                          <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                            <span>{getStatusIcon(task.status)}</span>
                            <span style={{ fontWeight: 600 }}>{task.schemaName}</span>
                          </div>
                        </td>
                        <td style={{ padding: '12px', textAlign: 'center' }}>
                          <span style={{ 
                            color: getStatusColor(task.status),
                            fontWeight: 600,
                            textTransform: 'capitalize'
                          }}>
                            {task.status}
                          </span>
                        </td>
                        <td style={{ padding: '12px', textAlign: 'center' }}>
                          {task.execution?.duration ? 
                            `${Math.floor(task.execution.duration / 60)}m ${task.execution.duration % 60}s` : 
                            '-'
                          }
                        </td>
                        <td style={{ padding: '12px', textAlign: 'center' }}>
                          {task.execution?.recordsGenerated?.toLocaleString() || task.sampleSize.toLocaleString()}
                        </td>
                        <td style={{ padding: '12px', textAlign: 'center' }}>
                          ${task.execution?.cost?.toFixed(3) || '0.000'}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </div>
        )}
      </div>

      <style>{`
        @keyframes pulse {
          0%, 100% { opacity: 1; }
          50% { opacity: 0.5; }
        }
        
        @keyframes progress {
          0% { transform: translateX(-100%); }
          50% { transform: translateX(0%); }
          100% { transform: translateX(100%); }
        }
      `}</style>
    </div>
  )
}

function formatRelativeTime(date: Date): string {
  const now = new Date()
  const diff = now.getTime() - date.getTime()
  const minutes = Math.floor(diff / (1000 * 60))
  const hours = Math.floor(diff / (1000 * 60 * 60))
  const days = Math.floor(diff / (1000 * 60 * 60 * 24))
  
  if (minutes < 60) return `${minutes}m ago`
  if (hours < 24) return `${hours}h ago`
  return `${days}d ago`
}
