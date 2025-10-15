// Airflow-inspired types for data generation pipelines

export type TaskStatus = 
  | 'none'           // Never run
  | 'scheduled'      // Scheduled to run
  | 'queued'         // In queue
  | 'running'        // Currently executing
  | 'success'        // Completed successfully
  | 'failed'         // Failed execution
  | 'up_for_retry'   // Failed but will retry
  | 'skipped'        // Skipped due to conditions
  | 'upstream_failed' // Failed due to upstream failure

export interface LogEntry {
  timestamp: Date
  level: 'INFO' | 'WARN' | 'ERROR' | 'DEBUG'
  message: string
  taskId?: string
}

export interface TaskExecution {
  runId: string
  taskId: string
  status: TaskStatus
  startTime?: Date
  endTime?: Date
  duration?: number
  recordsGenerated: number
  cost: number
  tryNumber: number
  maxTries: number
  logs: LogEntry[]
  error?: string
}

export interface GenerationTask {
  id: string
  schemaId: string
  schemaName: string
  sampleSize: number
  status: TaskStatus
  dependencies: string[] // Other task IDs this depends on
  position: { x: number; y: number } // For DAG visualization
  config: {
    model: string
    temperature: number
    customPrompt?: string
    batchSize: number
    maxRetries: number
    seed?: number
  }
  execution?: TaskExecution
}

export interface PipelineRun {
  id: string
  projectId: string
  status: TaskStatus
  startTime: Date
  endTime?: Date
  duration?: number
  totalRecords: number
  totalCost: number
  tasks: GenerationTask[]
  triggeredBy: 'manual' | 'schedule' | 'api'
}

export interface SchemaProject {
  id: string
  name: string
  description: string
  owner: string
  schemaGroupId: string
  isActive: boolean
  schedule: 'manual' | 'daily' | 'weekly' | 'monthly' | string // cron expression
  
  // DAG configuration
  tasks: GenerationTask[]
  
  // Run history and stats
  lastRun?: PipelineRun
  stats: {
    totalRuns: number
    successRate: number
    avgDuration: number
    totalRecords: number
    totalCost: number
  }
  
  // Configuration
  defaultSettings: {
    model: string
    temperature: number
    batchSize: number
    maxParallel: number
  }
  
  notifications: {
    onSuccess: string[]
    onFailure: string[]
    channels: ('email' | 'slack' | 'webhook')[]
  }
  
  created: Date
  lastModified: Date
}

export interface ProjectsListFilter {
  search: string
  owner: string
  status: TaskStatus | 'all'
  schedule: string | 'all'
  sortBy: 'name' | 'lastRun' | 'successRate' | 'totalRuns'
  sortOrder: 'asc' | 'desc'
}

// DAG View modes (like Airflow)
export type DAGViewMode = 'graph' | 'tree' | 'gantt' | 'details' | 'code'

// Task action types
export type TaskAction = 
  | 'run'
  | 'retry' 
  | 'skip'
  | 'mark_success'
  | 'mark_failed'
  | 'clear'
  | 'edit'
  | 'logs'
  | 'details'
