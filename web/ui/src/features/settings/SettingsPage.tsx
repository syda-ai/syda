import { useState } from 'react'
import { useTheme } from '../../store/ThemeContext'

type SettingsTab = 'profile' | 'appearance' | 'ai-models' | 'notifications' | 'security' | 'usage' | 'about'

interface ProviderConfig {
  name: string
  key: string
  apiKey: string
  status: 'connected' | 'testing' | 'error' | 'not-configured'
  extraKwargs?: Record<string, string>  // All provider-specific configuration (base_url, organization_id, etc.)
  lastTested?: string
}

export default function SettingsPage() {
  const { theme, setTheme } = useTheme()
  const [activeTab, setActiveTab] = useState<SettingsTab>('appearance')
  const [selectedProvider, setSelectedProvider] = useState<string>('')
  
  // Provider configurations
  const [providers, setProviders] = useState<Record<string, ProviderConfig>>({
    anthropic: {
      name: 'Anthropic',
      key: 'anthropic',
      apiKey: 'sk-ant-api03-**********************abc123',
      status: 'connected',
      extraKwargs: {},
      lastTested: '2 minutes ago'
    },
    openai: {
      name: 'OpenAI',
      key: 'openai',
      apiKey: '',
      status: 'not-configured',
      extraKwargs: {}
    },
    gemini: {
      name: 'Google Gemini',
      key: 'gemini',
      apiKey: 'AIza**********************xyz789',
      status: 'testing',
      extraKwargs: { 'project_id': 'my-project' }
    },
    azureopenai: {
      name: 'Azure OpenAI',
      key: 'azureopenai',
      apiKey: '',
      status: 'not-configured',
      extraKwargs: {}
    },
    grok: {
      name: 'Grok (xAI)',
      key: 'grok',
      apiKey: '',
      status: 'not-configured',
      extraKwargs: {}
    }
  })

  const settingsTabs = [
    { id: 'profile', label: 'Profile', icon: '👤' },
    { id: 'appearance', label: 'Appearance', icon: '🎨' },
    { id: 'ai-models', label: 'AI Models', icon: '🤖' },
    { id: 'notifications', label: 'Notifications', icon: '🔔' },
    { id: 'security', label: 'Security', icon: '🔒' },
    { id: 'usage', label: 'Usage & Billing', icon: '📊' },
    { id: 'about', label: 'About', icon: 'ℹ️' }
  ] as const

  // Helper functions
  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'connected': return '✅'
      case 'testing': return '🔄'
      case 'error': return '❌'
      case 'not-configured': return '⚠️'
      default: return '⚪'
    }
  }

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'connected': return 'var(--success)'
      case 'testing': return 'var(--warn)'
      case 'error': return 'var(--danger)'
      case 'not-configured': return 'var(--warn)'
      default: return 'var(--muted)'
    }
  }

  const getStatusLabel = (status: string) => {
    switch (status) {
      case 'connected': return 'Connected'
      case 'testing': return 'Testing...'
      case 'error': return 'Connection Failed'
      case 'not-configured': return 'Not Configured'
      default: return 'Unknown'
    }
  }

  // Get provider stats
  const providerList = Object.values(providers)
  const totalProviders = providerList.length
  const configuredProviders = providerList.filter(p => p.status === 'connected').length
  const pendingProviders = providerList.filter(p => p.status === 'not-configured' || p.status === 'error').length

  // Get currently selected provider details
  const currentProvider = selectedProvider ? providers[selectedProvider] : null

  // Handler functions
  const updateProviderConfig = (providerKey: string, updates: Partial<ProviderConfig>) => {
    setProviders(prev => ({
      ...prev,
      [providerKey]: { ...prev[providerKey], ...updates }
    }))
  }

  const testConnection = async (providerKey: string) => {
    updateProviderConfig(providerKey, { status: 'testing' })
    
    // Simulate API test
    await new Promise(resolve => setTimeout(resolve, 2000))
    
    // Mock result based on API key presence
    const hasKey = providers[providerKey].apiKey
    updateProviderConfig(providerKey, { 
      status: hasKey ? 'connected' : 'error',
      lastTested: 'Just now'
    })
  }

  // Extra kwargs management
  const addExtraKwarg = (providerKey: string) => {
    const current = providers[providerKey].extraKwargs || {}
    updateProviderConfig(providerKey, {
      extraKwargs: { ...current, '': '' }
    })
  }

  const updateExtraKwarg = (providerKey: string, oldKey: string, newKey: string, value: string) => {
    const current = providers[providerKey].extraKwargs || {}
    const updated = { ...current }
    
    // Remove old key if it exists and key changed
    if (oldKey && oldKey !== newKey) {
      delete updated[oldKey]
    }
    
    // Add/update new key
    if (newKey) {
      updated[newKey] = value
    }
    
    updateProviderConfig(providerKey, { extraKwargs: updated })
  }

  const removeExtraKwarg = (providerKey: string, key: string) => {
    const current = providers[providerKey].extraKwargs || {}
    const updated = { ...current }
    delete updated[key]
    updateProviderConfig(providerKey, { extraKwargs: updated })
  }

  return (
    <div className="settings-layout">
      {/* GitHub-style Sidebar */}
      <div className="settings-sidebar">
        <div className="header">
          <h2 style={{ margin: 0, fontSize: '1.25rem', fontWeight: 700 }}>
            ⚙️ Settings
          </h2>
          <p className="subtitle">
            Manage your account and application preferences
          </p>
        </div>

        <nav className="settings-tabs">
          {settingsTabs.map(tab => (
            <button
              key={tab.id}
              onClick={() => setActiveTab(tab.id as SettingsTab)}
              className={`settings-tab ${activeTab === tab.id ? 'active' : ''}`}
            >
              <span>{tab.icon}</span>
              {tab.label}
            </button>
          ))}
        </nav>
      </div>

      {/* Main Content */}
      <div className="settings-content">
        {activeTab === 'appearance' && (
          <div style={{ maxWidth: 800 }}>
            <div style={{ marginBottom: 32 }}>
              <h3 style={{ margin: '0 0 8px 0', fontSize: '1.5rem', fontWeight: 700 }}>
                🎨 Appearance
              </h3>
              <p style={{ margin: 0, color: 'var(--muted)', fontSize: '1rem' }}>
                Customize how Syda UI looks and feels
              </p>
            </div>

            <div className="panel" style={{ padding: 24, marginBottom: 24 }}>
              <h4 style={{ margin: '0 0 16px 0', fontSize: '1.1rem', fontWeight: 600 }}>
                Theme preference
              </h4>
              <p style={{ margin: '0 0 20px 0', color: 'var(--muted)', fontSize: '0.9rem' }}>
                Choose between light and dark themes
              </p>

              <div style={{ display: 'grid', gap: 16 }}>
                <label style={{ 
                  display: 'flex', 
                  alignItems: 'center', 
                  gap: 16, 
                  cursor: 'pointer',
                  padding: 16,
                  border: theme === 'dark' ? '2px solid var(--primary)' : '2px solid var(--border)',
                  borderRadius: 12,
                  background: theme === 'dark' ? 'rgba(59, 130, 246, 0.05)' : 'transparent'
                }}>
                  <input 
                    type="radio" 
                    name="theme" 
                    value="dark" 
                    checked={theme === 'dark'} 
                    onChange={() => setTheme('dark')}
                    style={{ accentColor: 'var(--primary)' }}
                  />
                  <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
                    <span style={{ fontSize: '1.5rem' }}>🌙</span>
                    <div>
                      <div style={{ fontWeight: 600, fontSize: '1rem' }}>Dark Theme</div>
                      <div style={{ color: 'var(--muted)', fontSize: '0.9rem' }}>
                        Easy on the eyes in low light conditions
                      </div>
                    </div>
                  </div>
                </label>

                <label style={{ 
                  display: 'flex', 
                  alignItems: 'center', 
                  gap: 16, 
                  cursor: 'pointer',
                  padding: 16,
                  border: theme === 'light' ? '2px solid var(--primary)' : '2px solid var(--border)',
                  borderRadius: 12,
                  background: theme === 'light' ? 'rgba(59, 130, 246, 0.05)' : 'transparent'
                }}>
                  <input 
                    type="radio" 
                    name="theme" 
                    value="light" 
                    checked={theme === 'light'} 
                    onChange={() => setTheme('light')}
                    style={{ accentColor: 'var(--primary)' }}
                  />
                  <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
                    <span style={{ fontSize: '1.5rem' }}>☀️</span>
                    <div>
                      <div style={{ fontWeight: 600, fontSize: '1rem' }}>Light Theme</div>
                      <div style={{ color: 'var(--muted)', fontSize: '0.9rem' }}>
                        Clean and bright interface
                      </div>
                    </div>
                  </div>
                </label>
              </div>
            </div>
          </div>
        )}

        {activeTab === 'ai-models' && (
          <div style={{ maxWidth: 800 }}>
            <div style={{ marginBottom: 32 }}>
              <h3 style={{ margin: '0 0 8px 0', fontSize: '1.5rem', fontWeight: 700 }}>
                🤖 AI Provider Configuration
              </h3>
              <p style={{ margin: 0, color: 'var(--muted)', fontSize: '1rem' }}>
                Configure API keys and credentials for AI providers. Model selection happens at runtime when generating data.
              </p>
            </div>

            {/* Quick Stats */}
            <div style={{ 
              display: 'grid', 
              gridTemplateColumns: 'repeat(auto-fit, minmax(180px, 1fr))', 
              gap: 16, 
              marginBottom: 24 
            }}>
              <div className="panel" style={{ padding: 16 }}>
                <div style={{ fontSize: '0.85rem', color: 'var(--muted)', marginBottom: 4 }}>
                  Total Providers
                </div>
                <div style={{ fontSize: '1.8rem', fontWeight: 700 }}>
                  {totalProviders}
                </div>
                <div style={{ fontSize: '0.75rem', color: 'var(--muted)', marginTop: 4 }}>
                  Available integrations
                </div>
              </div>
              
              <div className="panel" style={{ padding: 16 }}>
                <div style={{ fontSize: '0.85rem', color: 'var(--muted)', marginBottom: 4 }}>
                  Configured
                </div>
                <div style={{ fontSize: '1.8rem', fontWeight: 700, color: 'var(--success)' }}>
                  {configuredProviders}
                </div>
                <div style={{ fontSize: '0.75rem', color: 'var(--success)', marginTop: 4 }}>
                  ✅ Ready to use
                </div>
              </div>
              
              <div className="panel" style={{ padding: 16 }}>
                <div style={{ fontSize: '0.85rem', color: 'var(--muted)', marginBottom: 4 }}>
                  Pending Setup
                </div>
                <div style={{ fontSize: '1.8rem', fontWeight: 700, color: 'var(--warn)' }}>
                  {pendingProviders}
                </div>
                <div style={{ fontSize: '0.75rem', color: 'var(--warn)', marginTop: 4 }}>
                  ⚠️ Need API keys
                </div>
              </div>
            </div>

            {/* Provider Selector Dropdown */}
            <div className="panel" style={{ padding: 24, marginBottom: 24 }}>
              <h4 style={{ margin: '0 0 16px 0', fontSize: '1.1rem', fontWeight: 600 }}>
                🎯 Select Provider to Configure
              </h4>
              
              <div style={{ display: 'grid', gap: 16 }}>
                <div>
                  <label style={{ display: 'block', marginBottom: 8, fontWeight: 600, fontSize: '0.9rem' }}>
                    Choose an AI provider
                  </label>
                  <select 
                    className="select" 
                    style={{ width: '100%', fontSize: '1rem', padding: '12px' }}
                    value={selectedProvider}
                    onChange={(e) => setSelectedProvider(e.target.value)}
                  >
                    <option value="">-- Select a provider to configure --</option>
                    {providerList.map(provider => (
                      <option key={provider.key} value={provider.key}>
                        {provider.name} {provider.status === 'connected' ? '✅' : '⚠️'}
                      </option>
                    ))}
                  </select>
                  
                  <div style={{ fontSize: '0.8rem', color: 'var(--muted)', marginTop: 8 }}>
                    💡 Tip: Configure API keys here. Choose specific models at generation time.
                    <br />
                    ✅ = Configured  |  ⚠️ = Needs Configuration
                  </div>
                </div>
              </div>
            </div>

            {/* Dynamic Provider Configuration Panel */}
            {currentProvider && (
              <div className="panel" style={{ padding: 24, marginBottom: 24 }}>
                <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 20 }}>
                  <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
                    <h4 style={{ margin: 0, fontSize: '1.2rem', fontWeight: 600 }}>
                      {currentProvider.name}
                    </h4>
                    <div style={{ 
                      display: 'flex', 
                      alignItems: 'center', 
                      gap: 6,
                      padding: '4px 8px',
                      background: `${getStatusColor(currentProvider.status)}20`,
                      color: getStatusColor(currentProvider.status),
                      borderRadius: 6,
                      fontSize: '0.8rem',
                      fontWeight: 600
                    }}>
                      <span>{getStatusIcon(currentProvider.status)}</span>
                      {getStatusLabel(currentProvider.status)}
                    </div>
                  </div>
                  
                  {currentProvider.lastTested && (
                    <div style={{ fontSize: '0.85rem', color: 'var(--muted)' }}>
                      Last tested: {currentProvider.lastTested}
                    </div>
                  )}
                </div>

                <div style={{ display: 'grid', gap: 16 }}>
                  {/* API Key - Only required field */}
                  <div>
                    <label style={{ display: 'block', marginBottom: 8, fontWeight: 600, fontSize: '0.9rem' }}>
                      API Key <span style={{ color: 'var(--danger)' }}>*</span>
                    </label>
                    <input
                      className="input"
                      type="password"
                      value={currentProvider.apiKey}
                      onChange={(e) => updateProviderConfig(selectedProvider, { apiKey: e.target.value })}
                      placeholder={`Enter ${currentProvider.name} API key...`}
                      style={{ fontFamily: 'ui-monospace, monospace' }}
                    />
                    <div style={{ fontSize: '0.8rem', color: 'var(--muted)', marginTop: 4 }}>
                      {selectedProvider === 'anthropic' && (
                        <>Get your API key from <a href="https://console.anthropic.com" target="_blank" rel="noopener noreferrer" style={{ color: 'var(--primary)' }}>console.anthropic.com</a></>
                      )}
                      {selectedProvider === 'openai' && (
                        <>Get your API key from <a href="https://platform.openai.com/api-keys" target="_blank" rel="noopener noreferrer" style={{ color: 'var(--primary)' }}>platform.openai.com</a></>
                      )}
                      {selectedProvider === 'gemini' && (
                        <>Get your API key from <a href="https://makersuite.google.com/app/apikey" target="_blank" rel="noopener noreferrer" style={{ color: 'var(--primary)' }}>Google AI Studio</a></>
                      )}
                      {selectedProvider === 'grok' && (
                        <>Get your API key from <a href="https://x.ai" target="_blank" rel="noopener noreferrer" style={{ color: 'var(--primary)' }}>x.ai</a></>
                      )}
                      {selectedProvider === 'azureopenai' && (
                        <>Your Azure OpenAI API key from Azure Portal</>
                      )}
                    </div>
                  </div>

                  {/* Extra Kwargs - All Other Configuration */}
                  <details style={{ marginTop: 8 }}>
                    <summary style={{ 
                      cursor: 'pointer', 
                      padding: '12px 16px',
                      background: 'var(--hover-bg)',
                      border: '1px solid var(--border)',
                      borderRadius: 8,
                      fontWeight: 600,
                      fontSize: '0.9rem',
                      userSelect: 'none'
                    }}>
                      ⚙️ Advanced Configuration (Extra Kwargs)
                    </summary>
                    
                    <div style={{ 
                      marginTop: 12,
                      padding: 16,
                      background: 'var(--hover-bg)',
                      border: '1px solid var(--border)',
                      borderRadius: 8
                    }}>
                      <div style={{ marginBottom: 12, fontSize: '0.85rem', color: 'var(--muted)', lineHeight: 1.6 }}>
                        <strong>📝 What are Extra Kwargs?</strong>
                        <br />
                        Additional provider-specific parameters passed to the client during initialization. 
                        All configuration except the API key goes here as key-value pairs.
                        <br /><br />
                        <strong>💡 Common Examples for {currentProvider.name}:</strong>
                        {selectedProvider === 'anthropic' && (
                          <ul style={{ margin: '8px 0', paddingLeft: 20, fontSize: '0.9rem' }}>
                            <li><code>base_url</code>: Custom API endpoint (e.g., "https://api.anthropic.com" for proxy)</li>
                            <li><code>timeout</code>: Request timeout in seconds (e.g., "60")</li>
                            <li><code>max_retries</code>: Number of retry attempts (e.g., "3")</li>
                            <li><code>default_headers</code>: Custom HTTP headers as JSON string</li>
                            <li><code>http_client</code>: Custom HTTP client configuration</li>
                          </ul>
                        )}
                        {selectedProvider === 'openai' && (
                          <ul style={{ margin: '8px 0', paddingLeft: 20, fontSize: '0.9rem' }}>
                            <li><code>base_url</code>: Custom API endpoint (e.g., "https://api.openai.com/v1")</li>
                            <li><code>organization</code>: OpenAI organization ID (e.g., "org-xxxxxxxx")</li>
                            <li><code>timeout</code>: Request timeout in seconds (e.g., "30")</li>
                            <li><code>max_retries</code>: Number of retry attempts (e.g., "3")</li>
                            <li><code>default_headers</code>: Custom HTTP headers</li>
                            <li><code>http_client</code>: Custom HTTP client</li>
                          </ul>
                        )}
                        {selectedProvider === 'azureopenai' && (
                          <ul style={{ margin: '8px 0', paddingLeft: 20, fontSize: '0.9rem' }}>
                            <li><code>azure_endpoint</code>: Your Azure resource endpoint (e.g., "https://YOUR-RESOURCE.openai.azure.com") <strong>*Required</strong></li>
                            <li><code>api_version</code>: Azure API version (e.g., "2024-02-01")</li>
                            <li><code>azure_deployment</code>: Deployment name (e.g., "gpt-4-deployment")</li>
                            <li><code>azure_ad_token</code>: Azure AD authentication token</li>
                            <li><code>azure_ad_token_provider</code>: Token provider function</li>
                            <li><code>timeout</code>: Request timeout in seconds</li>
                            <li><code>max_retries</code>: Retry attempts</li>
                          </ul>
                        )}
                        {selectedProvider === 'gemini' && (
                          <ul style={{ margin: '8px 0', paddingLeft: 20, fontSize: '0.9rem' }}>
                            <li><code>project_id</code>: Google Cloud project ID (e.g., "my-project-123")</li>
                            <li><code>base_url</code>: Custom API endpoint</li>
                            <li><code>transport</code>: Custom transport configuration</li>
                            <li><code>client_options</code>: Additional client options</li>
                            <li><code>request_options</code>: Per-request options</li>
                          </ul>
                        )}
                        {selectedProvider === 'grok' && (
                          <ul style={{ margin: '8px 0', paddingLeft: 20, fontSize: '0.9rem' }}>
                            <li><code>base_url</code>: API endpoint (e.g., "https://api.x.ai")</li>
                            <li><code>timeout</code>: Request timeout in seconds (e.g., "30")</li>
                            <li><code>max_retries</code>: Number of retry attempts (e.g., "3")</li>
                            <li><code>default_headers</code>: Custom HTTP headers</li>
                          </ul>
                        )}
                        <div style={{ marginTop: 12, padding: 8, background: 'rgba(59, 130, 246, 0.1)', borderRadius: 6, fontSize: '0.85rem' }}>
                          💡 <strong>Tip:</strong> These parameters are passed directly to the provider's client initialization. 
                          Check the provider's official documentation for all available options.
                        </div>
                      </div>

                      {/* Key-Value Pairs */}
                      <div style={{ display: 'grid', gap: 12 }}>
                        {Object.entries(currentProvider.extraKwargs || {}).map(([key, value], idx) => (
                          <div key={idx} style={{ display: 'flex', gap: 8, alignItems: 'start' }}>
                            <div style={{ flex: 1 }}>
                              <input
                                className="input"
                                placeholder="Key (e.g., timeout)"
                                value={key}
                                onChange={(e) => updateExtraKwarg(selectedProvider, key, e.target.value, value)}
                                style={{ fontFamily: 'ui-monospace, monospace', fontSize: '0.9rem' }}
                              />
                            </div>
                            <div style={{ flex: 1 }}>
                              <input
                                className="input"
                                placeholder="Value (e.g., 30)"
                                value={value}
                                onChange={(e) => updateExtraKwarg(selectedProvider, key, key, e.target.value)}
                                style={{ fontFamily: 'ui-monospace, monospace', fontSize: '0.9rem' }}
                              />
                            </div>
                            <button
                              className="btn secondary"
                              onClick={() => removeExtraKwarg(selectedProvider, key)}
                              style={{ padding: '8px 12px', minWidth: 'auto' }}
                              title="Remove this parameter"
                            >
                              ❌
                            </button>
                          </div>
                        ))}
                        
                        {Object.keys(currentProvider.extraKwargs || {}).length === 0 && (
                          <div style={{ 
                            textAlign: 'center', 
                            padding: 20, 
                            color: 'var(--muted)', 
                            fontSize: '0.9rem',
                            border: '2px dashed var(--border)',
                            borderRadius: 8
                          }}>
                            No extra parameters configured. Click "Add Parameter" to add custom configuration.
                          </div>
                        )}

                        <button
                          className="btn secondary"
                          onClick={() => addExtraKwarg(selectedProvider)}
                          style={{ width: '100%' }}
                        >
                          ➕ Add Parameter
                        </button>
                      </div>
                    </div>
                  </details>

                  {/* Save & Test Button */}
                  <div style={{ display: 'flex', gap: 8 }}>
                    <button 
                      className="btn"
                      onClick={() => testConnection(selectedProvider)}
                      disabled={!currentProvider.apiKey || currentProvider.status === 'testing'}
                    >
                      {currentProvider.status === 'testing' ? '🔄 Testing...' : '💾 Save & Test'}
                    </button>
                    {currentProvider.status === 'connected' && (
                      <button 
                        className="btn secondary"
                        onClick={() => updateProviderConfig(selectedProvider, { apiKey: '', extraKwargs: {}, status: 'not-configured' })}
                      >
                        ❌ Remove Configuration
                      </button>
                    )}
                  </div>
                </div>
              </div>
            )}

            {/* Empty State - When no provider is selected */}
            {!selectedProvider && (
              <div className="panel" style={{ padding: 40, textAlign: 'center' }}>
                <div style={{ fontSize: '3rem', marginBottom: 16 }}>🎯</div>
                <h4 style={{ margin: '0 0 8px 0', fontSize: '1.1rem' }}>
                  Select a provider to get started
                </h4>
                <p style={{ margin: 0, color: 'var(--muted)', fontSize: '0.9rem' }}>
                  Choose an AI provider from the dropdown above to configure its settings
                </p>
              </div>
            )}

            {/* Info Box */}
            <div style={{ 
              padding: 16, 
              background: 'var(--hover-bg)', 
              border: '1px solid var(--border)', 
              borderRadius: 12,
              fontSize: '0.9rem'
            }}>
              <div style={{ fontWeight: 600, marginBottom: 8 }}>💡 About Model Selection</div>
              <p style={{ margin: 0, color: 'var(--muted)', lineHeight: 1.6 }}>
                Provider configuration is done here in Settings. When generating synthetic data, you'll choose the specific model 
                (e.g., claude-3-5-sonnet, gpt-4, gemini-1.5-pro) and parameters (temperature, max_tokens, etc.) at runtime. 
                This gives you flexibility to try different models without reconfiguring API keys.
              </p>
            </div>
          </div>
        )}

        {activeTab === 'usage' && (
          <div style={{ maxWidth: 800 }}>
            <div style={{ marginBottom: 32 }}>
              <h3 style={{ margin: '0 0 8px 0', fontSize: '1.5rem', fontWeight: 700 }}>
                📊 Usage & Billing
              </h3>
              <p style={{ margin: 0, color: 'var(--muted)', fontSize: '1rem' }}>
                Monitor your AI model usage and costs
              </p>
            </div>

            <div className="panel" style={{ padding: 24, marginBottom: 24 }}>
              <h4 style={{ margin: '0 0 16px 0', fontSize: '1.1rem', fontWeight: 600 }}>
                💰 This month's usage
              </h4>
              
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: 20, marginBottom: 20 }}>
                <div>
                  <div style={{ fontSize: '0.8rem', color: 'var(--muted)', marginBottom: 4 }}>Total Spent</div>
                  <div style={{ fontSize: '2rem', fontWeight: 700, color: 'var(--primary)' }}>$23.45</div>
                </div>
                <div>
                  <div style={{ fontSize: '0.8rem', color: 'var(--muted)', marginBottom: 4 }}>Budget Limit</div>
                  <div style={{ fontSize: '2rem', fontWeight: 700 }}>$100.00</div>
                </div>
                <div>
                  <div style={{ fontSize: '0.8rem', color: 'var(--muted)', marginBottom: 4 }}>Records Generated</div>
                  <div style={{ fontSize: '2rem', fontWeight: 700, color: 'var(--success)' }}>127K</div>
                </div>
              </div>

              <div style={{ marginBottom: 20 }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 8 }}>
                  <span style={{ fontSize: '0.9rem', fontWeight: 600 }}>Budget usage</span>
                  <span style={{ fontSize: '0.9rem', color: 'var(--muted)' }}>23% of $100.00</span>
                </div>
                <div style={{ 
                  width: '100%', 
                  height: 8, 
                  background: 'var(--border)', 
                  borderRadius: 4,
                  overflow: 'hidden'
                }}>
                  <div style={{ 
                    width: '23%', 
                    height: '100%', 
                    background: 'linear-gradient(90deg, var(--success), var(--primary))',
                    transition: 'width 0.3s ease'
                  }} />
                </div>
              </div>

              <div style={{ display: 'flex', gap: 8 }}>
                <button className="btn secondary">📊 View detailed usage</button>
                <button className="btn secondary">📧 Setup budget alerts</button>
              </div>
            </div>

            {/* Usage by Provider */}
            <div className="panel" style={{ padding: 24 }}>
              <h4 style={{ margin: '0 0 16px 0', fontSize: '1.1rem', fontWeight: 600 }}>
                📈 Usage by provider
              </h4>
              
              <table style={{ width: '100%', borderCollapse: 'collapse' }}>
                <thead>
                  <tr style={{ borderBottom: '2px solid var(--border)' }}>
                    <th style={{ padding: '12px', textAlign: 'left', fontWeight: 600 }}>Provider</th>
                    <th style={{ padding: '12px', textAlign: 'right', fontWeight: 600 }}>Requests</th>
                    <th style={{ padding: '12px', textAlign: 'right', fontWeight: 600 }}>Cost</th>
                    <th style={{ padding: '12px', textAlign: 'right', fontWeight: 600 }}>Avg Response</th>
                  </tr>
                </thead>
                <tbody>
                  <tr style={{ borderBottom: '1px solid var(--border)' }}>
                    <td style={{ padding: '12px' }}>Anthropic (Claude)</td>
                    <td style={{ padding: '12px', textAlign: 'right' }}>1,247</td>
                    <td style={{ padding: '12px', textAlign: 'right', color: 'var(--primary)' }}>$18.23</td>
                    <td style={{ padding: '12px', textAlign: 'right' }}>1.2s</td>
                  </tr>
                  <tr style={{ borderBottom: '1px solid var(--border)' }}>
                    <td style={{ padding: '12px' }}>OpenAI (GPT)</td>
                    <td style={{ padding: '12px', textAlign: 'right' }}>342</td>
                    <td style={{ padding: '12px', textAlign: 'right', color: 'var(--primary)' }}>$5.22</td>
                    <td style={{ padding: '12px', textAlign: 'right' }}>0.8s</td>
                  </tr>
                  <tr>
                    <td style={{ padding: '12px' }}>Google (Gemini)</td>
                    <td style={{ padding: '12px', textAlign: 'right' }}>89</td>
                    <td style={{ padding: '12px', textAlign: 'right', color: 'var(--primary)' }}>$0.67</td>
                    <td style={{ padding: '12px', textAlign: 'right' }}>0.6s</td>
                  </tr>
                </tbody>
              </table>
            </div>
          </div>
        )}

        {activeTab === 'about' && (
          <div style={{ maxWidth: 800 }}>
            <div style={{ marginBottom: 32 }}>
              <h3 style={{ margin: '0 0 8px 0', fontSize: '1.5rem', fontWeight: 700 }}>
                ℹ️ About Syda UI
              </h3>
              <p style={{ margin: 0, color: 'var(--muted)', fontSize: '1rem' }}>
                Information about this application
              </p>
            </div>

            <div className="panel" style={{ padding: 24 }}>
              <div style={{ display: 'grid', gap: 16, fontSize: '0.95rem' }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', padding: '8px 0', borderBottom: '1px solid var(--border)' }}>
                  <span style={{ fontWeight: 600 }}>Version</span>
                  <span>1.0.0</span>
                </div>
                <div style={{ display: 'flex', justifyContent: 'space-between', padding: '8px 0', borderBottom: '1px solid var(--border)' }}>
                  <span style={{ fontWeight: 600 }}>Build</span>
                  <span>dev</span>
                </div>
                <div style={{ display: 'flex', justifyContent: 'space-between', padding: '8px 0', borderBottom: '1px solid var(--border)' }}>
                  <span style={{ fontWeight: 600 }}>Framework</span>
                  <span>React + TypeScript</span>
                </div>
                <div style={{ display: 'flex', justifyContent: 'space-between', padding: '8px 0' }}>
                  <span style={{ fontWeight: 600 }}>Last Updated</span>
                  <span>2024-01-15</span>
                </div>
              </div>
            </div>
          </div>
        )}

        {/* Placeholder for other tabs */}
        {!['appearance', 'ai-models', 'about', 'usage'].includes(activeTab) && (
          <div style={{ maxWidth: 800 }}>
            <div style={{ marginBottom: 32 }}>
              <h3 style={{ margin: '0 0 8px 0', fontSize: '1.5rem', fontWeight: 700 }}>
                {settingsTabs.find(t => t.id === activeTab)?.icon} {settingsTabs.find(t => t.id === activeTab)?.label}
              </h3>
              <p style={{ margin: 0, color: 'var(--muted)', fontSize: '1rem' }}>
                Coming soon...
              </p>
            </div>

            <div className="panel" style={{ padding: 40, textAlign: 'center' }}>
              <div style={{ fontSize: '3rem', marginBottom: 16 }}>🚧</div>
              <h4 style={{ margin: '0 0 8px 0' }}>Under Construction</h4>
              <p style={{ margin: 0, color: 'var(--muted)' }}>
                This settings section is coming soon.
              </p>
            </div>
          </div>
        )}
      </div>
    </div>
  )
}
