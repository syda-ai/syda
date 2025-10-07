// Syda-compliant AI model configuration types and utilities

export type ModelProvider = 'openai' | 'anthropic' | 'gemini'

export interface ProxyConfig {
  http_proxy?: string
  https_proxy?: string
  no_proxy?: string
}

export interface SydaModelConfig {
  // Core model settings
  provider: ModelProvider
  model_name: string
  
  // Common parameters
  temperature?: number // 0.0 - 1.0
  max_tokens?: number
  
  // Sampling parameters
  top_k?: number
  top_p?: number // 0.0 - 1.0
  
  // Streaming
  stream?: boolean
  
  // OpenAI specific
  seed?: number
  response_format?: Record<string, any>
  max_completion_tokens?: number
  
  // Anthropic specific  
  max_tokens_to_sample?: number
  
  // Proxy configuration
  proxy?: ProxyConfig
  
  // Custom generation settings (not part of Syda core)
  customPrompt?: string
  batchSize?: number
  maxRetries?: number
}

// Available models by provider
export const AVAILABLE_MODELS = {
  openai: [
    { value: 'gpt-4o', label: 'GPT-4o', costPer1K: 0.35, speed: 15, quality: 0.95 },
    { value: 'gpt-4o-mini', label: 'GPT-4o Mini', costPer1K: 0.15, speed: 25, quality: 0.88 },
    { value: 'gpt-4-turbo', label: 'GPT-4 Turbo', costPer1K: 0.30, speed: 18, quality: 0.93 },
    { value: 'gpt-3.5-turbo', label: 'GPT-3.5 Turbo', costPer1K: 0.08, speed: 35, quality: 0.82 }
  ],
  anthropic: [
    { value: 'claude-3-5-sonnet-20241022', label: 'Claude 3.5 Sonnet', costPer1K: 0.18, speed: 22, quality: 0.96 },
    { value: 'claude-3-5-haiku-20241022', label: 'Claude 3.5 Haiku', costPer1K: 0.08, speed: 45, quality: 0.85 },
    { value: 'claude-3-opus-20240229', label: 'Claude 3 Opus', costPer1K: 0.45, speed: 12, quality: 0.98 }
  ],
  gemini: [
    { value: 'gemini-1.5-pro', label: 'Gemini 1.5 Pro', costPer1K: 0.12, speed: 28, quality: 0.89 },
    { value: 'gemini-1.5-flash', label: 'Gemini 1.5 Flash', costPer1K: 0.06, speed: 50, quality: 0.83 }
  ]
} as const

// Default model configurations
export const DEFAULT_MODEL_CONFIGS: Record<string, SydaModelConfig> = {
  'cost-optimized': {
    provider: 'anthropic',
    model_name: 'claude-3-5-haiku-20241022',
    temperature: 0.7,
    max_tokens: 4000,
    batchSize: 150,
    maxRetries: 3,
    customPrompt: 'Generate realistic, diverse data efficiently while maintaining quality.'
  },
  'quality-focused': {
    provider: 'anthropic',
    model_name: 'claude-3-5-sonnet-20241022',
    temperature: 0.8,
    max_tokens: 8000,
    batchSize: 50,
    maxRetries: 5,
    customPrompt: 'Generate high-quality, detailed, and creative data with rich relationships and realistic patterns.'
  },
  'speed-optimized': {
    provider: 'gemini',
    model_name: 'gemini-1.5-flash',
    temperature: 0.6,
    max_tokens: 2000,
    batchSize: 200,
    maxRetries: 2,
    customPrompt: 'Generate data quickly while maintaining basic quality and consistency.'
  },
  'balanced': {
    provider: 'anthropic',
    model_name: 'claude-3-5-sonnet-20241022',
    temperature: 0.7,
    max_tokens: 4000,
    batchSize: 100,
    maxRetries: 3,
    customPrompt: 'Generate realistic data with good balance of quality, speed, and cost.'
  }
}

// Get model info by provider and model name
export function getModelInfo(provider: ModelProvider, modelName: string) {
  const models = AVAILABLE_MODELS[provider]
  return models.find(m => m.value === modelName) || models[0]
}

// Calculate cost estimate for a schema group
export function calculateGroupCostEstimate(
  modelConfig: SydaModelConfig, 
  totalRecords: number
): { cost: number; duration: number; recordsPerSec: number } {
  const modelInfo = getModelInfo(modelConfig.provider, modelConfig.model_name)
  
  const cost = (totalRecords / 1000) * modelInfo.costPer1K
  const recordsPerSec = modelInfo.speed * (modelConfig.batchSize || 100) / 100 // Adjust for batch size
  const duration = totalRecords / recordsPerSec
  
  return {
    cost: Math.round(cost * 100) / 100, // Round to 2 decimal places
    duration: Math.round(duration),
    recordsPerSec: Math.round(recordsPerSec * 10) / 10
  }
}

// Get model strategy recommendations
export function getModelStrategyRecommendations(
  schemaCount: number, 
  totalRecords: number,
  hasComplexSchemas: boolean
): Array<{
  id: string
  name: string
  description: string
  config: SydaModelConfig
  estimate: ReturnType<typeof calculateGroupCostEstimate>
}> {
  const strategies = [
    {
      id: 'cost-optimized',
      name: '💰 Cost Optimized',
      description: 'Best value for money with good quality',
      config: DEFAULT_MODEL_CONFIGS['cost-optimized']
    },
    {
      id: 'balanced',
      name: '⚖️ Balanced',
      description: 'Good balance of cost, speed, and quality',
      config: DEFAULT_MODEL_CONFIGS['balanced']
    },
    {
      id: 'quality-focused',
      name: '🎯 Quality Focused',
      description: 'Highest quality output, premium cost',
      config: DEFAULT_MODEL_CONFIGS['quality-focused']
    },
    {
      id: 'speed-optimized',
      name: '⚡ Speed Optimized',
      description: 'Fastest generation with acceptable quality',
      config: DEFAULT_MODEL_CONFIGS['speed-optimized']
    }
  ]

  return strategies.map(strategy => ({
    ...strategy,
    estimate: calculateGroupCostEstimate(strategy.config, totalRecords)
  }))
}

// Validate model configuration
export function validateModelConfig(config: SydaModelConfig): string[] {
  const errors: string[] = []
  
  if (!config.provider) {
    errors.push('Provider is required')
  }
  
  if (!config.model_name) {
    errors.push('Model name is required')
  }
  
  if (config.temperature !== undefined && (config.temperature < 0 || config.temperature > 1)) {
    errors.push('Temperature must be between 0.0 and 1.0')
  }
  
  if (config.top_p !== undefined && (config.top_p < 0 || config.top_p > 1)) {
    errors.push('Top P must be between 0.0 and 1.0')
  }
  
  if (config.max_tokens !== undefined && config.max_tokens < 1) {
    errors.push('Max tokens must be positive')
  }
  
  if (config.batchSize !== undefined && config.batchSize < 1) {
    errors.push('Batch size must be positive')
  }
  
  return errors
}

// Get provider-specific parameter visibility
export function getProviderParameters(provider: ModelProvider): Array<keyof SydaModelConfig> {
  const common: Array<keyof SydaModelConfig> = ['temperature', 'max_tokens', 'customPrompt', 'batchSize', 'maxRetries']
  
  switch (provider) {
    case 'openai':
      return [...common, 'seed', 'response_format', 'max_completion_tokens', 'top_p']
    
    case 'anthropic':
      return [...common, 'max_tokens_to_sample', 'top_k', 'top_p']
    
    case 'gemini':
      return [...common, 'top_k', 'top_p', 'stream']
    
    default:
      return common
  }
}

// Format model name for display
export function formatModelName(provider: ModelProvider, modelName: string): string {
  const modelInfo = getModelInfo(provider, modelName)
  return modelInfo.label
}

// Get model recommendation based on schema complexity
export function getModelRecommendation(
  schemaCount: number,
  avgFieldsPerSchema: number,
  hasComplexRelationships: boolean,
  prioritizeCost: boolean = false
): keyof typeof DEFAULT_MODEL_CONFIGS {
  
  if (prioritizeCost) {
    return 'cost-optimized'
  }
  
  if (hasComplexRelationships || avgFieldsPerSchema > 15) {
    return 'quality-focused'
  }
  
  if (schemaCount > 10 || avgFieldsPerSchema < 5) {
    return 'speed-optimized'
  }
  
  return 'balanced'
}
