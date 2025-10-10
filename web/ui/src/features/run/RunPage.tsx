import { useState } from 'react'
import ProjectsList from './ProjectsList'
import DAGView from './DAGView'
import TaskDetailPanel from './TaskDetailPanel'
import type { SchemaProject, TaskAction } from './types'

type RunPageView = 'projects' | 'dag' | 'task'

export default function RunPage() {
  const [currentView, setCurrentView] = useState<RunPageView>('projects')
  const [selectedProject, setSelectedProject] = useState<SchemaProject | null>(null)
  const [selectedTask, setSelectedTask] = useState<string | null>(null)
  const [showLogsModal, setShowLogsModal] = useState(false)

  const buildMockProject = (projectId: string): SchemaProject => ({
      id: projectId,
      name: 'E-commerce Pipeline',
      description: 'Data generation pipeline for e-commerce schemas',
      owner: 'Admin',
      schemaGroupId: 'ecommerce',
      isActive: true,
      schedule: 'manual',
      tasks: [
        {
          id: 'task_users',
          schemaId: 'users',
          schemaName: 'Users',
          sampleSize: 1000,
          status: 'success',
          dependencies: [],
          position: { x: 0, y: 100 },
          config: {
            model: 'Claude Sonnet 3.5',
            temperature: 0.7,
            batchSize: 100,
            maxRetries: 3,
            customPrompt: 'Generate diverse user profiles with realistic names and demographics'
          },
          execution: {
            runId: 'run_123',
            taskId: 'task_users',
            status: 'success',
            startTime: new Date(Date.now() - 2 * 60 * 60 * 1000),
            endTime: new Date(Date.now() - 2 * 60 * 60 * 1000 + 45 * 1000),
            duration: 45,
            recordsGenerated: 1000,
            cost: 0.15,
            tryNumber: 1,
            maxTries: 3,
            logs: [
              { timestamp: new Date(Date.now() - 2 * 60 * 60 * 1000), level: 'INFO', message: 'Task started', taskId: 'task_users' },
              { timestamp: new Date(Date.now() - 2 * 60 * 60 * 1000 + 10 * 1000), level: 'INFO', message: 'Generating records...', taskId: 'task_users' },
              { timestamp: new Date(Date.now() - 2 * 60 * 60 * 1000 + 45 * 1000), level: 'INFO', message: 'Task completed successfully', taskId: 'task_users' }
            ]
          }
        },
        {
          id: 'task_categories',
          schemaId: 'categories',
          schemaName: 'Categories',
          sampleSize: 50,
          status: 'success',
          dependencies: [],
          position: { x: 0, y: 200 },
          config: {
            model: 'Claude Sonnet 3.5',
            temperature: 0.7,
            batchSize: 25,
            maxRetries: 3
          },
          execution: {
            runId: 'run_123',
            taskId: 'task_categories',
            status: 'success',
            startTime: new Date(Date.now() - 2 * 60 * 60 * 1000 + 45 * 1000),
            endTime: new Date(Date.now() - 2 * 60 * 60 * 1000 + 57 * 1000),
            duration: 12,
            recordsGenerated: 50,
            cost: 0.05,
            tryNumber: 1,
            maxTries: 3,
            logs: [
              { timestamp: new Date(Date.now() - 2 * 60 * 60 * 1000 + 45 * 1000), level: 'INFO', message: 'Task started', taskId: 'task_categories' },
              { timestamp: new Date(Date.now() - 2 * 60 * 60 * 1000 + 57 * 1000), level: 'INFO', message: 'Task completed successfully', taskId: 'task_categories' }
            ]
          }
        },
        {
          id: 'task_products',
          schemaId: 'products',
          schemaName: 'Products',
          sampleSize: 500,
          status: 'running',
          dependencies: ['task_users', 'task_categories'],
          position: { x: 250, y: 150 },
          config: {
            model: 'Claude Sonnet 3.5',
            temperature: 0.7,
            batchSize: 50,
            maxRetries: 3,
            customPrompt: 'Generate realistic product data with proper category associations'
          },
          execution: {
            runId: 'run_124',
            taskId: 'task_products',
            status: 'running',
            startTime: new Date(Date.now() - 60 * 1000),
            recordsGenerated: 300,
            cost: 0.09,
            tryNumber: 1,
            maxTries: 3,
            logs: [
              { timestamp: new Date(Date.now() - 60 * 1000), level: 'INFO', message: 'Task started', taskId: 'task_products' },
              { timestamp: new Date(Date.now() - 30 * 1000), level: 'INFO', message: 'Generating records (60%)...', taskId: 'task_products' }
            ]
          }
        },
        {
          id: 'task_orders',
          schemaId: 'orders',
          schemaName: 'Orders',
          sampleSize: 2000,
          status: 'queued',
          dependencies: ['task_users', 'task_products'],
          position: { x: 500, y: 100 },
          config: {
            model: 'Claude Sonnet 3.5',
            temperature: 0.7,
            batchSize: 100,
            maxRetries: 3
          }
        },
        {
          id: 'task_reviews',
          schemaId: 'reviews',
          schemaName: 'Reviews',
          sampleSize: 500,
          status: 'queued',
          dependencies: ['task_users', 'task_products'],
          position: { x: 500, y: 200 },
          config: {
            model: 'Claude Sonnet 3.5',
            temperature: 0.8,
            batchSize: 50,
            maxRetries: 3
          }
        }
      ],
      stats: {
        totalRuns: 12,
        successRate: 0.92,
        avgDuration: 135,
        totalRecords: 4050,
        totalCost: 2.45
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
      created: new Date(Date.now() - 30 * 24 * 60 * 60 * 1000),
      lastModified: new Date(Date.now() - 2 * 60 * 60 * 1000)
    })

  const handleSelectProject = (projectId: string) => {
    const mockProject = buildMockProject(projectId)
    setSelectedProject(mockProject)
    setCurrentView('dag')
  }

  const handleTaskAction = (taskId: string, action: TaskAction) => {
    switch (action) {
      case 'details':
      case 'edit':
        setSelectedTask(taskId)
        setCurrentView('task')
        break
      case 'logs':
        setSelectedTask(taskId)
        setShowLogsModal(true)
        break
      case 'run':
        // TODO: Trigger individual task
        console.log(`Running task: ${taskId}`)
        break
      case 'retry':
        // TODO: Retry failed task
        console.log(`Retrying task: ${taskId}`)
        break
      default:
        console.log(`Task action: ${action} on ${taskId}`)
    }
  }

  const handleProjectAction = (action: string) => {
    switch (action) {
      case 'trigger':
        // TODO: Start entire pipeline
        console.log('Triggering DAG')
        break
      case 'pause':
        // TODO: Pause pipeline
        console.log('Pausing DAG')
        break
      case 'refresh':
        // TODO: Refresh status
        console.log('Refreshing DAG')
        break
      default:
        console.log(`Project action: ${action}`)
    }
  }

  const handleCreateProject = () => {
    // TODO: Open project creation wizard
    console.log('Creating new project')
  }

  const handleBackToProjects = () => {
    setCurrentView('projects')
    setSelectedProject(null)
    setSelectedTask(null)
  }

  const handleBackToDAG = () => {
    setCurrentView('dag')
    setSelectedTask(null)
  }

  const currentTask = selectedProject?.tasks.find(t => t.id === selectedTask)

  // (reverted) project-level logs quick-open helper removed

  return (
    <div className="page-container" style={{ height: '100%', width: '100%', minWidth: 0, display: 'flex', flexDirection: 'column', gap: 0 }}>
      {/* Breadcrumb Navigation */}
      {currentView !== 'projects' && (
        <div style={{ 
          padding: '12px 20px', 
          borderBottom: '1px solid var(--border)',
          background: 'var(--panel-2)',
          display: 'flex',
          alignItems: 'center',
          gap: 8,
          fontSize: '0.9rem'
        }}>
          <button 
            className="btn secondary" 
            onClick={handleBackToProjects}
            style={{ padding: '4px 8px', fontSize: '0.8rem' }}
          >
            ⚙️ Projects
          </button>
          {selectedProject && (
            <>
              <span style={{ color: 'var(--muted)' }}>&gt;</span>
              <button 
                className="btn secondary" 
                onClick={handleBackToDAG}
                style={{ padding: '4px 8px', fontSize: '0.8rem' }}
              >
                📊 {selectedProject.name}
              </button>
            </>
          )}
          {currentTask && (
            <>
              <span style={{ color: 'var(--muted)' }}>&gt;</span>
              <span style={{ fontWeight: 600 }}>{currentTask.schemaName}</span>
            </>
          )}
        </div>
      )}

      {/* Main Content */}
      <div style={{ flex: 1, overflow: 'hidden' }}>
        {currentView === 'projects' && (
          <ProjectsList
            onSelectProject={handleSelectProject}
            onCreateProject={handleCreateProject}
          />
        )}

        {currentView === 'dag' && selectedProject && (
          <DAGView
            project={selectedProject}
            onTaskAction={handleTaskAction}
            onProjectAction={handleProjectAction}
          />
        )}

        {currentView === 'task' && currentTask && (
          <TaskDetailPanel
            task={currentTask}
            onAction={(action) => handleTaskAction(currentTask.id, action)}
            onClose={handleBackToDAG}
          />
        )}
      </div>

      {/* Logs Modal */}
      {showLogsModal && currentTask && (
        <div style={{
          position: 'fixed', top: 0, left: 0, right: 0, bottom: 0,
          background: 'rgba(0,0,0,0.5)', display: 'flex', alignItems: 'center', justifyContent: 'center', zIndex: 1000
        }}>
          <div className="panel" style={{ width: 800, maxWidth: '95vw', maxHeight: '80vh', display: 'flex', flexDirection: 'column' }}>
            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', padding: 16, borderBottom: '1px solid var(--border)' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                <span>📋</span>
                <strong>Logs — {currentTask.schemaName}</strong>
              </div>
              <button className="btn secondary" onClick={() => setShowLogsModal(false)} style={{ padding: '4px 8px', fontSize: '0.8rem' }}>✕</button>
            </div>
            <div style={{ padding: 16, overflow: 'auto', flex: 1 }}>
              {currentTask.execution?.logs && currentTask.execution.logs.length > 0 ? (
                <pre style={{ margin: 0, fontSize: 12, lineHeight: 1.5 }}>
{currentTask.execution.logs
  .map((l: any) => {
    const ts = l.timestamp instanceof Date ? l.timestamp : new Date(l.timestamp)
    return `[${ts.toLocaleString()}] ${l.level}: ${l.message}`
  })
  .join('\n')}
                </pre>
              ) : (
                <div style={{ color: 'var(--muted)' }}>No logs available yet.</div>
              )}
            </div>
            <div style={{ padding: 12, borderTop: '1px solid var(--border)', display: 'flex', justifyContent: 'flex-end', gap: 8 }}>
              <button className="btn secondary" onClick={() => setShowLogsModal(false)}>Close</button>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}



