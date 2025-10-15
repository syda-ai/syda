import { useState, useMemo } from 'react'
import { useAppState } from '../../store/AppState'
import type { SchemaProject, ProjectsListFilter, TaskStatus } from './types'

type ProjectsListProps = {
  onSelectProject: (projectId: string) => void
  onCreateProject: () => void
}

// Mock data for demonstration
function createMockProjects(schemaGroups: any[]): SchemaProject[] {
  return schemaGroups.map(group => ({
    id: `project_${group.id}`,
    name: `${group.name} Pipeline`,
    description: `Data generation pipeline for ${group.name} schemas`,
    owner: 'Admin',
    schemaGroupId: group.id,
    isActive: true,
    schedule: group.id === 'ecommerce' ? 'manual' : 'daily',
    tasks: group.schemas.map((schema: any, index: number) => ({
      id: `task_${schema.id}`,
      schemaId: schema.id,
      schemaName: schema.name,
      sampleSize: schema.kind === 'template' ? 5 : (index === 0 ? 1000 : 500),
      status: (index === 0 ? 'success' : index === 1 ? 'running' : 'queued') as TaskStatus,
      dependencies: index === 0 ? [] : [`task_${group.schemas[index - 1].id}`],
      position: { x: index * 200, y: 100 },
      config: {
        model: 'Claude Sonnet 3.5',
        temperature: 0.7,
        batchSize: 100,
        maxRetries: 3,
        customPrompt: `Generate realistic ${schema.name.toLowerCase()} data`
      }
    })),
    lastRun: {
      id: `run_${Date.now()}`,
      projectId: `project_${group.id}`,
      status: 'success' as TaskStatus,
      startTime: new Date(Date.now() - 2 * 60 * 60 * 1000), // 2 hours ago
      endTime: new Date(Date.now() - 2 * 60 * 60 * 1000 + 2 * 60 * 1000), // 2 minutes duration
      duration: 2 * 60, // 2 minutes in seconds
      totalRecords: group.schemas.length * 500,
      totalCost: 0.45,
      tasks: [],
      triggeredBy: 'manual'
    },
    stats: {
      totalRuns: Math.floor(Math.random() * 20) + 5,
      successRate: 0.85 + Math.random() * 0.15,
      avgDuration: 120 + Math.random() * 180, // 2-5 minutes
      totalRecords: group.schemas.length * 1000,
      totalCost: Math.random() * 10 + 2
    },
    defaultSettings: {
      model: 'Claude Sonnet 3.5',
      temperature: 0.7,
      batchSize: 100,
      maxParallel: 4
    },
    notifications: {
      onSuccess: [],
      onFailure: [],
      channels: []
    },
    created: new Date(Date.now() - 30 * 24 * 60 * 60 * 1000), // 30 days ago
    lastModified: new Date(Date.now() - 2 * 60 * 60 * 1000) // 2 hours ago
  }))
}

function getStatusIcon(status: TaskStatus): string {
  switch (status) {
    case 'success': return '✅'
    case 'failed': return '❌'
    case 'running': return '🔄'
    case 'queued': return '⏳'
    case 'scheduled': return '📅'
    default: return '⚪'
  }
}

function getStatusColor(status: TaskStatus): string {
  switch (status) {
    case 'success': return 'var(--success)'
    case 'failed': return 'var(--danger)'
    case 'running': return 'var(--warn)'
    case 'queued': return 'var(--primary)'
    case 'scheduled': return 'var(--accent)'
    default: return 'var(--muted)'
  }
}

function formatDuration(seconds: number): string {
  if (seconds < 60) return `${seconds}s`
  if (seconds < 3600) return `${Math.floor(seconds / 60)}m ${seconds % 60}s`
  return `${Math.floor(seconds / 3600)}h ${Math.floor((seconds % 3600) / 60)}m`
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

export default function ProjectsList({ onSelectProject, onCreateProject }: ProjectsListProps) {
  const { schemaGroups } = useAppState()
  const [filter, setFilter] = useState<ProjectsListFilter>({
    search: '',
    owner: 'all',
    status: 'all',
    schedule: 'all',
    sortBy: 'lastRun',
    sortOrder: 'desc'
  })

  const projects = useMemo(() => createMockProjects(schemaGroups), [schemaGroups])

  const filteredProjects = useMemo(() => {
    let filtered = projects

    // Apply search filter
    if (filter.search) {
      filtered = filtered.filter(p => 
        p.name.toLowerCase().includes(filter.search.toLowerCase()) ||
        p.description.toLowerCase().includes(filter.search.toLowerCase())
      )
    }

    // Apply status filter
    if (filter.status !== 'all') {
      filtered = filtered.filter(p => p.lastRun?.status === filter.status)
    }

    // Apply schedule filter
    if (filter.schedule !== 'all') {
      filtered = filtered.filter(p => p.schedule === filter.schedule)
    }

    // Apply sorting
    filtered.sort((a, b) => {
      let aVal: any, bVal: any
      
      switch (filter.sortBy) {
        case 'name':
          aVal = a.name
          bVal = b.name
          break
        case 'lastRun':
          aVal = a.lastRun?.startTime?.getTime() || 0
          bVal = b.lastRun?.startTime?.getTime() || 0
          break
        case 'successRate':
          aVal = a.stats.successRate
          bVal = b.stats.successRate
          break
        case 'totalRuns':
          aVal = a.stats.totalRuns
          bVal = b.stats.totalRuns
          break
        default:
          return 0
      }

      if (filter.sortOrder === 'asc') {
        return aVal < bVal ? -1 : aVal > bVal ? 1 : 0
      } else {
        return aVal > bVal ? -1 : aVal < bVal ? 1 : 0
      }
    })

    return filtered
  }, [projects, filter])

  return (
    <div style={{ padding: 20, height: '100%', overflow: 'auto' }}>
      {/* Header */}
      <div style={{ marginBottom: 24 }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 16 }}>
          <h2 style={{ margin: 0, display: 'flex', alignItems: 'center', gap: 8 }}>
            ⚙️ Data Generation Projects
          </h2>
          <div style={{ display: 'flex', gap: 8 }}>
            <button className="btn secondary" onClick={() => window.location.reload()}>
              🔄 Refresh
            </button>
            <button className="btn" onClick={onCreateProject}>
              + New Project
            </button>
          </div>
        </div>

        {/* Filters */}
        <div style={{ display: 'flex', gap: 12, alignItems: 'center', flexWrap: 'wrap' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
            <span style={{ fontSize: '0.9rem', fontWeight: 600 }}>🔍</span>
            <input
              className="input"
              placeholder="Search projects..."
              value={filter.search}
              onChange={(e) => setFilter(prev => ({ ...prev, search: e.target.value }))}
              style={{ width: 200 }}
            />
          </div>
          
          <select
            className="select"
            value={filter.status}
            onChange={(e) => setFilter(prev => ({ ...prev, status: e.target.value as any }))}
          >
            <option value="all">All Status</option>
            <option value="success">Success</option>
            <option value="failed">Failed</option>
            <option value="running">Running</option>
            <option value="queued">Queued</option>
          </select>

          <select
            className="select"
            value={filter.schedule}
            onChange={(e) => setFilter(prev => ({ ...prev, schedule: e.target.value }))}
          >
            <option value="all">All Schedules</option>
            <option value="manual">Manual</option>
            <option value="daily">Daily</option>
            <option value="weekly">Weekly</option>
            <option value="monthly">Monthly</option>
          </select>

          <select
            className="select"
            value={`${filter.sortBy}_${filter.sortOrder}`}
            onChange={(e) => {
              const [sortBy, sortOrder] = e.target.value.split('_')
              setFilter(prev => ({ ...prev, sortBy: sortBy as any, sortOrder: sortOrder as any }))
            }}
          >
            <option value="lastRun_desc">Last Run (Recent)</option>
            <option value="lastRun_asc">Last Run (Oldest)</option>
            <option value="name_asc">Name (A-Z)</option>
            <option value="name_desc">Name (Z-A)</option>
            <option value="successRate_desc">Success Rate (High)</option>
            <option value="totalRuns_desc">Total Runs (Most)</option>
          </select>
        </div>
      </div>

      {/* Projects Table */}
      <div className="panel">
        <table style={{ width: '100%', borderCollapse: 'collapse' }}>
          <thead>
            <tr style={{ background: 'var(--panel-2)', borderBottom: '2px solid var(--border)' }}>
              <th style={{ padding: '16px', textAlign: 'left', fontWeight: 700 }}>
                Project (DAG)
              </th>
              <th style={{ padding: '16px', textAlign: 'left', fontWeight: 700 }}>
                Owner
              </th>
              <th style={{ padding: '16px', textAlign: 'center', fontWeight: 700 }}>
                Runs
              </th>
              <th style={{ padding: '16px', textAlign: 'center', fontWeight: 700 }}>
                Success Rate
              </th>
              <th style={{ padding: '16px', textAlign: 'left', fontWeight: 700 }}>
                Schedule
              </th>
              <th style={{ padding: '16px', textAlign: 'left', fontWeight: 700 }}>
                Last Run
              </th>
              <th style={{ padding: '16px', textAlign: 'center', fontWeight: 700 }}>
                Actions
              </th>
            </tr>
          </thead>
          <tbody>
            {filteredProjects.map(project => (
              <tr 
                key={project.id} 
                style={{ 
                  borderBottom: '1px solid var(--border)',
                  cursor: 'pointer',
                  transition: 'background 0.2s ease'
                }}
                onMouseEnter={(e) => e.currentTarget.style.background = 'rgba(59, 130, 246, 0.05)'}
                onMouseLeave={(e) => e.currentTarget.style.background = 'transparent'}
                onClick={() => onSelectProject(project.id)}
              >
                {/* Project Name & Info */}
                <td style={{ padding: '16px' }}>
                  <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
                    <div style={{ 
                      width: 12, 
                      height: 12, 
                      borderRadius: '50%', 
                      background: project.isActive ? 'var(--success)' : 'var(--muted)' 
                    }} />
                    <div>
                      <div style={{ fontWeight: 600, fontSize: '1rem', marginBottom: 4 }}>
                        {project.name}
                      </div>
                      <div style={{ fontSize: '0.85rem', color: 'var(--muted)' }}>
                        {project.tasks.length} schemas • {project.stats.totalRecords.toLocaleString()} total records
                      </div>
                    </div>
                  </div>
                </td>

                {/* Owner */}
                <td style={{ padding: '16px' }}>
                  <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                    <span>👤</span>
                    <span>{project.owner}</span>
                  </div>
                </td>

                {/* Total Runs */}
                <td style={{ padding: '16px', textAlign: 'center' }}>
                  <div style={{ fontWeight: 600, fontSize: '1.1rem' }}>
                    {project.stats.totalRuns}
                  </div>
                </td>

                {/* Success Rate */}
                <td style={{ padding: '16px', textAlign: 'center' }}>
                  <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 8 }}>
                    <div style={{ 
                      width: 40, 
                      height: 6, 
                      background: 'var(--border)', 
                      borderRadius: 3,
                      overflow: 'hidden'
                    }}>
                      <div style={{ 
                        width: `${project.stats.successRate * 100}%`, 
                        height: '100%', 
                        background: project.stats.successRate > 0.9 ? 'var(--success)' : 
                                   project.stats.successRate > 0.7 ? 'var(--warn)' : 'var(--danger)',
                        transition: 'width 0.3s ease'
                      }} />
                    </div>
                    <span style={{ fontSize: '0.9rem', fontWeight: 600 }}>
                      {Math.round(project.stats.successRate * 100)}%
                    </span>
                  </div>
                </td>

                {/* Schedule */}
                <td style={{ padding: '16px' }}>
                  <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                    <span>{project.schedule === 'manual' ? '👤' : '📅'}</span>
                    <span style={{ textTransform: 'capitalize' }}>{project.schedule}</span>
                  </div>
                </td>

                {/* Last Run */}
                <td style={{ padding: '16px' }}>
                  {project.lastRun ? (
                    <div>
                      <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 4 }}>
                        <span>{getStatusIcon(project.lastRun.status)}</span>
                        <span style={{ 
                          fontWeight: 600, 
                          color: getStatusColor(project.lastRun.status)
                        }}>
                          {project.lastRun.status.toUpperCase()}
                        </span>
                      </div>
                      <div style={{ fontSize: '0.85rem', color: 'var(--muted)' }}>
                        {formatRelativeTime(project.lastRun.startTime)} • {formatDuration(project.lastRun.duration || 0)}
                      </div>
                    </div>
                  ) : (
                    <span style={{ color: 'var(--muted)', fontSize: '0.9rem' }}>Never run</span>
                  )}
                </td>

                {/* Actions */}
                <td style={{ padding: '16px', textAlign: 'center' }}>
                  <div style={{ display: 'flex', gap: 4, justifyContent: 'center' }}>
                    <button
                      className="btn secondary"
                      onClick={(e) => {
                        e.stopPropagation()
                        // TODO: Trigger pipeline
                      }}
                      style={{ padding: '4px 8px', fontSize: '0.8rem' }}
                      title="Trigger DAG"
                    >
                      ▶️
                    </button>
                    <button
                      className="btn secondary"
                      onClick={(e) => {
                        e.stopPropagation()
                        // TODO: Open settings
                      }}
                      style={{ padding: '4px 8px', fontSize: '0.8rem' }}
                      title="Settings"
                    >
                      ⚙️
                    </button>
                    <button
                      className="btn secondary"
                      onClick={(e) => {
                        e.stopPropagation()
                        // TODO: View logs
                      }}
                      style={{ padding: '4px 8px', fontSize: '0.8rem' }}
                      title="Logs"
                    >
                      📋
                    </button>
                  </div>
                </td>
              </tr>
            ))}
          </tbody>
        </table>

        {filteredProjects.length === 0 && (
          <div style={{ padding: 60, textAlign: 'center', color: 'var(--muted)' }}>
            <div style={{ fontSize: '3rem', marginBottom: 16 }}>⚙️</div>
            <h3 style={{ margin: '0 0 8px 0' }}>No Projects Found</h3>
            <p style={{ marginBottom: 16 }}>
              {filter.search || filter.status !== 'all' || filter.schedule !== 'all' 
                ? 'No projects match your current filters.' 
                : 'Create your first data generation project to get started.'}
            </p>
            <button className="btn" onClick={onCreateProject}>
              + Create Project
            </button>
          </div>
        )}
      </div>

      {/* Recent Activity */}
      <div style={{ marginTop: 24 }}>
        <h3 style={{ margin: '0 0 16px 0', display: 'flex', alignItems: 'center', gap: 8 }}>
          📈 Recent Activity
        </h3>
        <div className="panel" style={{ padding: 16 }}>
          {projects.slice(0, 5).map(project => (
            <div key={project.id} style={{ 
              display: 'flex', 
              alignItems: 'center', 
              gap: 12, 
              padding: '8px 0',
              borderBottom: '1px solid var(--border)'
            }}>
              <span>{getStatusIcon(project.lastRun?.status || 'none')}</span>
              <span style={{ flex: 1 }}>
                <strong>{project.name}</strong> {project.lastRun?.status || 'never run'}
              </span>
              <span style={{ fontSize: '0.85rem', color: 'var(--muted)' }}>
                {project.lastRun ? formatRelativeTime(project.lastRun.startTime) : 'Never'}
              </span>
            </div>
          ))}
        </div>
      </div>
    </div>
  )
}
