import { useState, useMemo } from 'react'
import type { SchemaGroup } from '../../store/AppState'
import { 
  AVAILABLE_MODELS, 
  DEFAULT_MODEL_CONFIGS,
  getModelInfo,
  calculateGroupCostEstimate,
  getModelStrategyRecommendations,
  validateModelConfig,
  getProviderParameters,
  formatModelName
} from './modelConfig'
import type { SydaModelConfig, ModelProvider } from './modelConfig'

type ModelConfigModalProps = {
  group: SchemaGroup
  onSave: (modelConfig: SydaModelConfig) => void
  onClose: () => void
}

export default function ModelConfigModal({ group, onSave, onClose }: ModelConfigModalProps) {
  const [config, setConfig] = useState<SydaModelConfig>(
    group.modelConfig || DEFAULT_MODEL_CONFIGS['balanced']
  )
  const [activeTab, setActiveTab] = useState<'basic' | 'advanced' | 'strategies'>('basic')
  const [errors, setErrors] = useState<string[]>([])

  // Calculate total records for cost estimation
  const totalRecords = useMemo(() => {
    return group.schemas.reduce((sum, schema) => sum + (schema.metadata?.fieldCount || 5) * 100, 0)
  }, [group.schemas])

  // Calculate cost and performance estimates
  const estimates = useMemo(() => {
    return calculateGroupCostEstimate(config, totalRecords)
  }, [config, totalRecords])

  // Get available models for selected provider
  const availableModels = AVAILABLE_MODELS[config.provider]

  // Get strategy recommendations
  const strategies = useMemo(() => {
    const avgFields = group.schemas.length > 0 
      ? group.schemas.reduce((sum, s) => sum + (s.metadata?.fieldCount || 5), 0) / group.schemas.length
      : 5
    const hasComplexSchemas = group.schemas.some(s => s.content.includes('foreign_key') || s.content.includes('array'))
    
    return getModelStrategyRecommendations(group.schemas.length, totalRecords, hasComplexSchemas)
  }, [group.schemas, totalRecords])

  const updateConfig = (updates: Partial<SydaModelConfig>) => {
    const newConfig = { ...config, ...updates }
    setConfig(newConfig)
    
    // Validate configuration
    const validationErrors = validateModelConfig(newConfig)
    setErrors(validationErrors)
  }

  const handleSave = () => {
    if (errors.length === 0) {
      // Update estimates before saving
      const finalConfig = {
        ...config,
        estimatedCostPer1000: estimates.cost / (totalRecords / 1000),
        estimatedSpeedRecordsPerSec: estimates.recordsPerSec
      }
      onSave(finalConfig)
      onClose()
    }
  }

  const applyStrategy = (strategyId: string) => {
    const strategy = DEFAULT_MODEL_CONFIGS[strategyId]
    if (strategy) {
      updateConfig(strategy)
    }
  }

  const visibleParameters = getProviderParameters(config.provider)

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
        maxWidth: 800, 
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
          <div>
            <h3 style={{ margin: '0 0 8px 0', display: 'flex', alignItems: 'center', gap: 8 }}>
              🤖 AI Model Configuration
            </h3>
            <div style={{ fontSize: '0.9rem', color: 'var(--muted)' }}>
              Group: <strong>{group.name}</strong> • {group.schemas.length} schemas
            </div>
          </div>
          
          {/* Cost Estimate */}
          <div className="panel" style={{ 
            padding: 12,
            background: 'rgba(59, 130, 246, 0.1)',
            border: '1px solid rgba(59, 130, 246, 0.2)'
          }}>
            <div style={{ fontSize: '0.8rem', color: 'var(--muted)', marginBottom: 4 }}>
              Estimated Cost
            </div>
            <div style={{ fontSize: '1.2rem', fontWeight: 700, color: 'var(--primary)' }}>
              ${estimates.cost.toFixed(2)}
            </div>
            <div style={{ fontSize: '0.75rem', color: 'var(--muted)' }}>
              {estimates.recordsPerSec.toFixed(1)} records/sec
            </div>
          </div>
        </div>

        {/* Tabs */}
        <div style={{ 
          display: 'flex', 
          borderBottom: '1px solid var(--border)',
          background: 'var(--panel-2)'
        }}>
          {(['basic', 'advanced', 'strategies'] as const).map(tab => (
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
              {tab === 'basic' && '🎯'} 
              {tab === 'advanced' && '⚙️'} 
              {tab === 'strategies' && '📋'} 
              {tab}
            </button>
          ))}
        </div>

        {/* Content */}
        <div style={{ flex: 1, overflow: 'auto', padding: 24 }}>
          {activeTab === 'basic' && (
            <div style={{ display: 'grid', gap: 24 }}>
              {/* Provider and Model Selection */}
              <div>
                <h4 style={{ margin: '0 0 16px 0' }}>🤖 Model Selection</h4>
                <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16 }}>
                  <div>
                    <label style={{ display: 'block', marginBottom: 8, fontWeight: 600 }}>
                      Provider
                    </label>
                    <select
                      className="select"
                      value={config.provider}
                      onChange={(e) => {
                        const provider = e.target.value as ModelProvider
                        const firstModel = AVAILABLE_MODELS[provider][0]
                        updateConfig({ 
                          provider, 
                          model_name: firstModel.value 
                        })
                      }}
                    >
                      <option value="anthropic">Anthropic (Claude)</option>
                      <option value="openai">OpenAI (GPT)</option>
                      <option value="gemini">Google (Gemini)</option>
                    </select>
                  </div>
                  
                  <div>
                    <label style={{ display: 'block', marginBottom: 8, fontWeight: 600 }}>
                      Model
                    </label>
                    <select
                      className="select"
                      value={config.model_name}
                      onChange={(e) => updateConfig({ model_name: e.target.value })}
                    >
                      {availableModels.map(model => (
                        <option key={model.value} value={model.value}>
                          {model.label} (${model.costPer1K}/1K)
                        </option>
                      ))}
                    </select>
                  </div>
                </div>
              </div>

              {/* Basic Parameters */}
              <div>
                <h4 style={{ margin: '0 0 16px 0' }}>⚙️ Generation Parameters</h4>
                <div style={{ display: 'grid', gap: 16 }}>
                  <div>
                    <label style={{ display: 'block', marginBottom: 8, fontWeight: 600 }}>
                      Temperature: {config.temperature || 0.7}
                    </label>
                    <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
                      <span style={{ fontSize: '0.8rem', color: 'var(--muted)' }}>Consistent</span>
                      <input
                        type="range"
                        min="0"
                        max="1"
                        step="0.1"
                        value={config.temperature || 0.7}
                        onChange={(e) => updateConfig({ temperature: parseFloat(e.target.value) })}
                        style={{ flex: 1 }}
                      />
                      <span style={{ fontSize: '0.8rem', color: 'var(--muted)' }}>Creative</span>
                    </div>
                  </div>

                  <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16 }}>
                    <div>
                      <label style={{ display: 'block', marginBottom: 8, fontWeight: 600 }}>
                        Max Tokens
                      </label>
                      <input
                        className="input"
                        type="number"
                        value={config.max_tokens || 4000}
                        onChange={(e) => updateConfig({ max_tokens: parseInt(e.target.value) })}
                        min="100"
                        max="32000"
                      />
                    </div>
                    
                    <div>
                      <label style={{ display: 'block', marginBottom: 8, fontWeight: 600 }}>
                        Batch Size
                      </label>
                      <input
                        className="input"
                        type="number"
                        value={config.batchSize || 100}
                        onChange={(e) => updateConfig({ batchSize: parseInt(e.target.value) })}
                        min="1"
                        max="500"
                      />
                    </div>
                  </div>
                </div>
              </div>

              {/* Custom Prompt */}
              <div>
                <h4 style={{ margin: '0 0 16px 0' }}>🎯 Custom Prompt</h4>
                <textarea
                  className="textarea"
                  value={config.customPrompt || ''}
                  onChange={(e) => updateConfig({ customPrompt: e.target.value })}
                  placeholder="Enter custom instructions for data generation..."
                  style={{ height: 100, resize: 'vertical' }}
                />
                <div style={{ fontSize: '0.8rem', color: 'var(--muted)', marginTop: 8 }}>
                  This prompt will be used for all schemas in the "{group.name}" group
                </div>
              </div>
            </div>
          )}

          {activeTab === 'advanced' && (
            <div style={{ display: 'grid', gap: 24 }}>
              {/* Provider-Specific Parameters */}
              <div>
                <h4 style={{ margin: '0 0 16px 0' }}>🔧 Advanced Parameters</h4>
                <div style={{ display: 'grid', gap: 16 }}>
                  {visibleParameters.includes('top_p') && (
                    <div>
                      <label style={{ display: 'block', marginBottom: 8, fontWeight: 600 }}>
                        Top P: {config.top_p || 0.9}
                      </label>
                      <input
                        type="range"
                        min="0"
                        max="1"
                        step="0.05"
                        value={config.top_p || 0.9}
                        onChange={(e) => updateConfig({ top_p: parseFloat(e.target.value) })}
                        style={{ width: '100%' }}
                      />
                      <div style={{ fontSize: '0.8rem', color: 'var(--muted)' }}>
                        Nucleus sampling parameter
                      </div>
                    </div>
                  )}

                  {visibleParameters.includes('top_k') && (
                    <div>
                      <label style={{ display: 'block', marginBottom: 8, fontWeight: 600 }}>
                        Top K
                      </label>
                      <input
                        className="input"
                        type="number"
                        value={config.top_k || 40}
                        onChange={(e) => updateConfig({ top_k: parseInt(e.target.value) })}
                        min="1"
                        max="100"
                      />
                      <div style={{ fontSize: '0.8rem', color: 'var(--muted)' }}>
                        Number of top tokens to consider
                      </div>
                    </div>
                  )}

                  {visibleParameters.includes('seed') && (
                    <div>
                      <label style={{ display: 'block', marginBottom: 8, fontWeight: 600 }}>
                        Seed (Optional)
                      </label>
                      <input
                        className="input"
                        type="number"
                        value={config.seed || ''}
                        onChange={(e) => updateConfig({ seed: e.target.value ? parseInt(e.target.value) : undefined })}
                        placeholder="For reproducible results"
                      />
                    </div>
                  )}

                  <div>
                    <label style={{ display: 'block', marginBottom: 8, fontWeight: 600 }}>
                      Max Retries
                    </label>
                    <input
                      className="input"
                      type="number"
                      value={config.maxRetries || 3}
                      onChange={(e) => updateConfig({ maxRetries: parseInt(e.target.value) })}
                      min="0"
                      max="10"
                    />
                  </div>

                  {visibleParameters.includes('stream') && (
                    <div>
                      <label style={{ display: 'flex', alignItems: 'center', gap: 8, cursor: 'pointer' }}>
                        <input
                          type="checkbox"
                          checked={config.stream || false}
                          onChange={(e) => updateConfig({ stream: e.target.checked })}
                        />
                        <div>
                          <div style={{ fontWeight: 600 }}>Enable Streaming</div>
                          <div style={{ fontSize: '0.8rem', color: 'var(--muted)' }}>
                            Stream responses for better user experience
                          </div>
                        </div>
                      </label>
                    </div>
                  )}
                </div>
              </div>
            </div>
          )}

          {activeTab === 'strategies' && (
            <div style={{ display: 'grid', gap: 16 }}>
              <div>
                <h4 style={{ margin: '0 0 16px 0' }}>📋 Quick Start Strategies</h4>
                <div style={{ fontSize: '0.9rem', color: 'var(--muted)', marginBottom: 16 }}>
                  Choose a pre-configured strategy optimized for different priorities
                </div>
              </div>

              {strategies.map(strategy => (
                <div key={strategy.id} className="panel" style={{ padding: 16 }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'start', marginBottom: 12 }}>
                    <div>
                      <h5 style={{ margin: '0 0 8px 0', display: 'flex', alignItems: 'center', gap: 8 }}>
                        {strategy.name}
                      </h5>
                      <p style={{ margin: 0, fontSize: '0.9rem', color: 'var(--muted)' }}>
                        {strategy.description}
                      </p>
                    </div>
                    <button
                      className="btn secondary"
                      onClick={() => applyStrategy(strategy.id)}
                      style={{ padding: '6px 12px', fontSize: '0.85rem' }}
                    >
                      Apply Strategy
                    </button>
                  </div>
                  
                  <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(120px, 1fr))', gap: 12, fontSize: '0.85rem' }}>
                    <div>
                      <div style={{ color: 'var(--muted)' }}>Model</div>
                      <div style={{ fontWeight: 600 }}>
                        {formatModelName(strategy.config.provider, strategy.config.model_name)}
                      </div>
                    </div>
                    <div>
                      <div style={{ color: 'var(--muted)' }}>Cost</div>
                      <div style={{ fontWeight: 600, color: 'var(--primary)' }}>
                        ${strategy.estimate.cost.toFixed(2)}
                      </div>
                    </div>
                    <div>
                      <div style={{ color: 'var(--muted)' }}>Speed</div>
                      <div style={{ fontWeight: 600 }}>
                        {strategy.estimate.recordsPerSec.toFixed(1)} r/s
                      </div>
                    </div>
                    <div>
                      <div style={{ color: 'var(--muted)' }}>Duration</div>
                      <div style={{ fontWeight: 600 }}>
                        {Math.floor(strategy.estimate.duration / 60)}m {strategy.estimate.duration % 60}s
                      </div>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>

        {/* Validation Errors */}
        {errors.length > 0 && (
          <div className="panel" style={{ 
            margin: '0 24px',
            padding: 12, 
            background: 'rgba(239, 68, 68, 0.1)', 
            border: '1px solid rgba(239, 68, 68, 0.2)' 
          }}>
            <div style={{ fontWeight: 600, marginBottom: 8, color: 'var(--danger)' }}>
              ⚠️ Configuration Issues:
            </div>
            <ul style={{ margin: 0, paddingLeft: 20, color: 'var(--danger)', fontSize: '0.9rem' }}>
              {errors.map((error, index) => (
                <li key={index}>{error}</li>
              ))}
            </ul>
          </div>
        )}

        {/* Footer */}
        <div style={{ 
          padding: 24, 
          borderTop: '1px solid var(--border)', 
          display: 'flex', 
          justifyContent: 'space-between',
          alignItems: 'center'
        }}>
          <div style={{ fontSize: '0.9rem', color: 'var(--muted)' }}>
            Configuration will apply to all {group.schemas.length} schemas in this group
          </div>
          
          <div style={{ display: 'flex', gap: 8 }}>
            <button className="btn secondary" onClick={onClose}>
              Cancel
            </button>
            <button 
              className="btn" 
              onClick={handleSave}
              disabled={errors.length > 0}
            >
              Apply Configuration
            </button>
          </div>
        </div>
      </div>
    </div>
  )
}
