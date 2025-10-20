import { useState } from 'react'
import { useTheme } from '../../store/ThemeContext'
import './SettingsPage.css'

type SettingsTab = 'profile' | 'appearance' | 'ai-models' | 'notifications' | 'security' | 'usage' | 'about'

interface APIKeyConfig {
  anthropic: string
  openai: string
  gemini: string
}

interface ConnectionStatus {
  anthropic: 'connected' | 'testing' | 'error' | 'not-configured'
  openai: 'connected' | 'testing' | 'error' | 'not-configured'
  gemini: 'connected' | 'testing' | 'error' | 'not-configured'
}

export default function SettingsPage() {
  const { theme, setTheme } = useTheme()
  const [activeTab, setActiveTab] = useState<SettingsTab>('appearance')
  const [apiKeys, setApiKeys] = useState<APIKeyConfig>({
    anthropic: 'sk-ant-api03-**********************abc123',
    openai: '',
    gemini: 'AIza**********************xyz789'
  })
  const [connectionStatus, setConnectionStatus] = useState<ConnectionStatus>({
    anthropic: 'connected',
    openai: 'not-configured',
    gemini: 'testing'
  })

  // Security tab state
  const [securityName, setSecurityName] = useState('')
  const [securityText, setSecurityText] = useState('')

  const settingsTabs = [
    { id: 'profile', label: 'Profile', icon: '👤' },
    { id: 'appearance', label: 'Appearance', icon: '🎨' },
    { id: 'ai-models', label: 'AI Models', icon: '🤖' },
    { id: 'notifications', label: 'Notifications', icon: '🔔' },
    { id: 'security', label: 'Security', icon: '🔒' },
    { id: 'usage', label: 'Usage & Billing', icon: '📊' },
    { id: 'about', label: 'About', icon: 'ℹ️' }
  ] as const

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

  const testConnection = async (provider: keyof APIKeyConfig) => {
    setConnectionStatus(prev => ({ ...prev, [provider]: 'testing' }))
    
    // Simulate API test
    await new Promise(resolve => setTimeout(resolve, 2000))
    
    // Mock result based on API key presence
    const hasKey = apiKeys[provider]
    setConnectionStatus(prev => ({ 
      ...prev, 
      [provider]: hasKey ? 'connected' : 'error' 
    }))
  }

  const updateApiKey = (provider: keyof APIKeyConfig, key: string) => {
    setApiKeys(prev => ({ ...prev, [provider]: key }))
    if (key) {
      setConnectionStatus(prev => ({ ...prev, [provider]: 'not-configured' }))
    }
  }

  return (
    <div className="settings-layout">
      {/* GitHub-style Sidebar */}
      <div className="settings-sidebar">
        <div className="header">
          <h2 className="settings-header-title">⚙️ Settings</h2>
          <p className="subtitle">Manage your account and application preferences</p>
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
          <div className="content-narrow">
            <div className="section-header">
              <h3 className="section-title">🎨 Appearance</h3>
              <p className="section-subtitle">Customize how Syda UI looks and feels</p>
            </div>

            <div className="panel panel--p-24 panel--mb-24">
              <h4 className="h4-title">Theme preference</h4>
              <p className="section-subtitle text-sm mb-20">Choose between light and dark themes</p>

              <div className="grid-gap-16">
                <label className={`theme-option ${theme === 'dark' ? 'active' : ''}`}>
                  <input 
                    type="radio" 
                    name="theme" 
                    value="dark" 
                    checked={theme === 'dark'} 
                    onChange={() => setTheme('dark')}
                    className="accent-primary"
                  />
                  <div className="hstack-12">
                    <span className="emoji-lg">🌙</span>
                    <div>
                      <div className="title-sm">Dark Theme</div>
                      <div className="text-muted text-sm">Easy on the eyes in low light conditions</div>
                    </div>
                  </div>
                </label>

                <label className={`theme-option ${theme === 'light' ? 'active' : ''}`}>
                  <input 
                    type="radio" 
                    name="theme" 
                    value="light" 
                    checked={theme === 'light'} 
                    onChange={() => setTheme('light')}
                    className="accent-primary"
                  />
                  <div className="hstack-12">
                    <span className="emoji-lg">☀️</span>
                    <div>
                      <div className="title-sm">Light Theme</div>
                      <div className="text-muted text-sm">Clean and bright interface</div>
                    </div>
                  </div>
                </label>
              </div>
            </div>
          </div>
        )}

        {activeTab === 'ai-models' && (
          <div className="content-narrow">
            <div className="section-header">
              <h3 className="section-title">🤖 AI Model Providers</h3>
              <p className="section-subtitle">Configure API keys and settings for AI providers used in data generation</p>
            </div>

            {/* Anthropic Configuration */}
            <div className="panel panel--p-24 panel--mb-24">
              <div className="panel-row">
                <div className="hstack-12">
                  <h4 className="h4-title" style={{ margin: 0, fontSize: '1.2rem' }}>
                    Anthropic (Claude)
                  </h4>
                  <div 
                    className="status-pill"
                    style={{ background: `${getStatusColor(connectionStatus.anthropic)}20`, color: getStatusColor(connectionStatus.anthropic) }}
                  >
                    <span>{getStatusIcon(connectionStatus.anthropic)}</span>
                    {connectionStatus.anthropic === 'connected' ? 'Connected' :
                     connectionStatus.anthropic === 'testing' ? 'Testing...' :
                     connectionStatus.anthropic === 'error' ? 'Connection Failed' :
                     'Not Configured'}
                  </div>
                </div>
                
                {connectionStatus.anthropic === 'connected' && (
                  <div className="text-xs text-muted">
                    Last tested: 2 minutes ago
                  </div>
                )}
              </div>

              <div className="grid-gap-16">
                <div>
                  <label className="field-label">Personal access token</label>
                  <input
                    className="input monospace"
                    type="password"
                    value={apiKeys.anthropic}
                    onChange={(e) => updateApiKey('anthropic', e.target.value)}
                    placeholder="sk-ant-api03-..."
                  />
                  <div className="text-sm text-muted mt-4">
                    Your Anthropic API key. Get one from{' '}
                    <a href="https://console.anthropic.com" target="_blank" rel="noopener noreferrer" className="link-primary">
                      console.anthropic.com
                    </a>
                  </div>
                </div>

                <div className="hstack-8">
                  <button 
                    className="btn secondary"
                    onClick={() => testConnection('anthropic')}
                    disabled={connectionStatus.anthropic === 'testing'}
                  >
                    {connectionStatus.anthropic === 'testing' ? '🔄 Testing...' : '🔍 Test connection'}
                  </button>
                  {connectionStatus.anthropic === 'connected' && (
                    <button className="btn secondary">
                      🔄 Regenerate token
                    </button>
                  )}
                </div>
              </div>
            </div>

            {/* OpenAI Configuration */}
            <div className="panel panel--p-24 panel--mb-24">
              <div className="panel-row">
                <div className="hstack-12">
                  <h4 className="h4-title" style={{ margin: 0, fontSize: '1.2rem' }}>
                    
                    OpenAI (GPT)
                  </h4>
                  <div 
                    className="status-pill"
                    style={{ background: `${getStatusColor(connectionStatus.openai)}20`, color: getStatusColor(connectionStatus.openai) }}
                  >
                    <span>{getStatusIcon(connectionStatus.openai)}</span>
                    API key required
                  </div>
                </div>
              </div>

              <div className="grid-gap-16">
                <div>
                  <label className="field-label">API key</label>
                  <input
                    className="input monospace"
                    type="password"
                    value={apiKeys.openai}
                    onChange={(e) => updateApiKey('openai', e.target.value)}
                    placeholder="sk-..."
                  />
                  <div className="text-sm text-muted mt-4">
                    Your OpenAI API key. Get one from{' '}
                    <a href="https://platform.openai.com/api-keys" target="_blank" rel="noopener noreferrer" className="link-primary">
                      platform.openai.com
                    </a>
                  </div>
                </div>

                <div>
                  <label className="field-label">Organization (optional)</label>
                  <input
                    className="input monospace"
                    placeholder="org-xxxxxxxxxxxxxxxx"
                  />
                </div>

                <button 
                  className="btn"
                  onClick={() => testConnection('openai')}
                  disabled={!apiKeys.openai || connectionStatus.openai === 'testing'}
                >
                  {connectionStatus.openai === 'testing' ? '🔄 Testing...' : '💾 Save and test'}
                </button>
              </div>
            </div>

            {/* Gemini Configuration */}
            <div className="panel panel--p-24 panel--mb-24">
              <div className="panel-row">
                <div className="hstack-12">
                  <h4 className="h4-title" style={{ margin: 0, fontSize: '1.2rem' }}>
                    Google (Gemini)
                  </h4>
                  <div 
                    className="status-pill"
                    style={{ background: `${getStatusColor(connectionStatus.gemini)}20`, color: getStatusColor(connectionStatus.gemini) }}
                  >
                    <span>{getStatusIcon(connectionStatus.gemini)}</span>
                    Testing connection...
                  </div>
                </div>
              </div>

              <div className="grid-gap-16">
                <div>
                  <label className="field-label">API key</label>
                  <input
                    className="input monospace"
                    type="password"
                    value={apiKeys.gemini}
                    onChange={(e) => updateApiKey('gemini', e.target.value)}
                    placeholder="AIza..."
                  />
                </div>

                <div>
                  <label className="field-label">Project ID</label>
                  <input
                    className="input monospace"
                    placeholder="my-project-123"
                  />
                </div>

                <button 
                  className="btn secondary"
                  onClick={() => testConnection('gemini')}
                  disabled={connectionStatus.gemini === 'testing'}
                >
                  🔍 Test connection
                </button>
              </div>
            </div>

            {/* Global Defaults */}
            <div className="panel panel--p-24">
              <h4 className="h4-title">🎯 Global defaults</h4>
              
              <div className="grid-gap-16">
                <div className="two-col-grid">
                  <div>
                    <label className="field-label">Default provider</label>
                    <select className="select">
                      <option value="anthropic">Anthropic (Claude)</option>
                      <option value="openai">OpenAI (GPT)</option>
                      <option value="gemini">Google (Gemini)</option>
                    </select>
                  </div>
                  
                  <div>
                    <label className="field-label">Fallback provider</label>
                    <select className="select">
                      <option value="anthropic">Anthropic (Claude)</option>
                      <option value="openai">OpenAI (GPT)</option>
                      <option value="gemini">Google (Gemini)</option>
                    </select>
                  </div>
                </div>

                <div className="two-col-grid">
                  <div>
                    <label className="field-label">Global timeout (seconds)</label>
                    <input className="input" type="number" defaultValue="30" />
                  </div>
                  
                  <div>
                    <label className="field-label">Global max retries</label>
                    <input className="input" type="number" defaultValue="3" />
                  </div>
                </div>
              </div>
            </div>
          </div>
        )}

        {activeTab === 'usage' && (
          <div className="content-narrow">
            <div className="section-header">
              <h3 className="section-title">📊 Usage & Billing</h3>
              <p className="section-subtitle">Monitor your AI model usage and costs</p>
            </div>

            <div className="panel panel--p-24 panel--mb-24">
              <h4 className="h4-title">💰 This month's usage</h4>
              
              <div className="autofit-200 mb-20">
                <div>
                  <div className="text-sm text-muted mb-4">Total Spent</div>
                  <div className="text-lg text-primary" style={{ fontWeight: 700 }}>$23.45</div>
                </div>
                <div>
                  <div className="text-sm text-muted mb-4">Budget Limit</div>
                  <div className="text-lg" style={{ fontWeight: 700 }}>$100.00</div>
                </div>
                <div>
                  <div className="text-sm text-muted mb-4">Records Generated</div>
                  <div className="text-lg text-success" style={{ fontWeight: 700 }}>127K</div>
                </div>
              </div>

              <div className="mb-20">
                <div className="panel-row" style={{ marginBottom: 8 }}>
                  <span className="text-sm" style={{ fontWeight: 600 }}>Budget usage</span>
                  <span className="text-sm text-muted">23% of $100.00</span>
                </div>
                <div className="progress">
                  <div className="progress-bar" style={{ width: '23%' }} />
                </div>
              </div>

              <div className="hstack-8">
                <button className="btn secondary">📊 View detailed usage</button>
                <button className="btn secondary">📧 Setup budget alerts</button>
              </div>
            </div>

            {/* Usage by Provider */}
            <div className="panel panel--p-24">
              <h4 className="h4-title">📈 Usage by provider</h4>
              
              <table className="table">
                <thead>
                  <tr>
                    <th>Provider</th>
                    <th className="text-right">Requests</th>
                    <th className="text-right">Cost</th>
                    <th className="text-right">Avg Response</th>
                  </tr>
                </thead>
                <tbody>
                  <tr>
                    <td>Anthropic (Claude)</td>
                    <td className="text-right">1,247</td>
                    <td className="text-right text-primary">$18.23</td>
                    <td className="text-right">1.2s</td>
                  </tr>
                  <tr>
                    <td>OpenAI (GPT)</td>
                    <td className="text-right">342</td>
                    <td className="text-right text-primary">$5.22</td>
                    <td className="text-right">0.8s</td>
                  </tr>
                  <tr>
                    <td>Google (Gemini)</td>
                    <td className="text-right">89</td>
                    <td className="text-right text-primary">$0.67</td>
                    <td className="text-right">0.6s</td>
                  </tr>
                </tbody>
              </table>
            </div>
          </div>
        )}

        {activeTab === 'about' && (
          <div className="content-narrow">
            <div className="section-header">
              <h3 className="section-title">ℹ️ About Syda UI</h3>
              <p className="section-subtitle">Information about this application</p>
            </div>

            <div className="panel panel--p-24">
              <div className="grid-gap-16 text-95">
                <div className="row-split">
                  <span style={{ fontWeight: 600 }}>Version</span>
                  <span>1.0.0</span>
                </div>
                <div className="row-split">
                  <span style={{ fontWeight: 600 }}>Build</span>
                  <span>dev</span>
                </div>
                <div className="row-split">
                  <span style={{ fontWeight: 600 }}>Framework</span>
                  <span>React + TypeScript</span>
                </div>
                <div className="row-split no-border">
                  <span style={{ fontWeight: 600 }}>Last Updated</span>
                  <span>2024-01-15</span>
                </div>
              </div>
            </div>
          </div>
        )}

        {activeTab === 'security' && (
          <div className="content-narrow">
            <div className="section-header">
              <h3 className="section-title">🔒 Security</h3>
            </div>

            <div className="panel panel--p-24">
              <div className="grid-gap-16">
                <div>
                  <label className="field-label">Name</label>
                  <input
                    className="input input-large"
                    placeholder="Enter name"
                    value={securityName}
                    onChange={(e) => setSecurityName(e.target.value)}
                  />
                </div>

                <div>
                  <label className="field-label">Security</label>
                  <textarea
                    className="input input-large"
                    rows={4}
                    placeholder="Enter security details"
                    value={securityText}
                    onChange={(e) => setSecurityText(e.target.value)}
                    style={{ maxHeight: 200, minHeight: 96, resize: 'vertical' }}
                  />
                </div>

                <div className="actions-right">
                  <button 
                    className="btn btn-sm"
                    onClick={() => console.log('Add security', { name: securityName, security: securityText })}
                    disabled={!securityName || !securityText}
                  >
                    Add security
                  </button>
                </div>
              </div>
            </div>
          </div>
        )}

        {/* Placeholder for other tabs */}
        {!['appearance', 'ai-models', 'about', 'usage', 'security'].includes(activeTab) && (
          <div className="content-narrow">
            <div className="section-header">
              <h3 className="section-title">
                {settingsTabs.find(t => t.id === activeTab)?.icon} {settingsTabs.find(t => t.id === activeTab)?.label}
              </h3>
            </div>

            <div className="panel panel--p-40 centered">
              <div className="emoji-3xl mb-16">🚧</div>
              <h4 className="h4-tight">Under Construction</h4>
              <p className="m-0 text-muted">This settings section is coming soon.</p>
            </div>
          </div>
        )}
      </div>
    </div>
  )
}
