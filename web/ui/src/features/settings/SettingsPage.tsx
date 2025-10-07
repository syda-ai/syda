import { useState } from 'react'
import { useTheme } from '../../store/ThemeContext'

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
    <div style={{ display: 'grid', gridTemplateColumns: '280px 1fr', height: '100vh', overflow: 'hidden' }}>
      {/* GitHub-style Sidebar */}
      <div style={{ 
        background: 'var(--panel-2)', 
        borderRight: '1px solid var(--border)',
        padding: '20px 0',
        overflow: 'auto'
      }}>
        <div style={{ padding: '0 20px 20px 20px' }}>
          <h2 style={{ margin: 0, fontSize: '1.25rem', fontWeight: 700 }}>
            ⚙️ Settings
          </h2>
          <p style={{ margin: '8px 0 0 0', fontSize: '0.9rem', color: 'var(--muted)' }}>
            Manage your account and application preferences
          </p>
        </div>

        <nav style={{ padding: '0 12px' }}>
          {settingsTabs.map(tab => (
            <button
              key={tab.id}
              onClick={() => setActiveTab(tab.id as SettingsTab)}
              style={{
                display: 'flex',
                alignItems: 'center',
                gap: 12,
                width: '100%',
                padding: '12px 16px',
                background: activeTab === tab.id ? 'var(--panel)' : 'transparent',
                border: activeTab === tab.id ? '1px solid var(--border)' : '1px solid transparent',
                borderRadius: 8,
                color: activeTab === tab.id ? 'var(--text)' : 'var(--muted)',
                fontSize: '0.9rem',
                fontWeight: activeTab === tab.id ? 600 : 400,
                cursor: 'pointer',
                transition: 'all 0.2s ease',
                marginBottom: 4,
                textAlign: 'left'
              }}
            >
              <span>{tab.icon}</span>
              {tab.label}
            </button>
          ))}
        </nav>
      </div>

      {/* Main Content */}
      <div style={{ overflow: 'auto', padding: 40 }}>
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
                🤖 AI Model Providers
              </h3>
              <p style={{ margin: 0, color: 'var(--muted)', fontSize: '1rem' }}>
                Configure API keys and settings for AI providers used in data generation
              </p>
            </div>

            {/* Anthropic Configuration */}
            <div className="panel" style={{ padding: 24, marginBottom: 24 }}>
              <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 20 }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
                  <h4 style={{ margin: 0, fontSize: '1.2rem', fontWeight: 600 }}>
                    Anthropic (Claude)
                  </h4>
                  <div style={{ 
                    display: 'flex', 
                    alignItems: 'center', 
                    gap: 6,
                    padding: '4px 8px',
                    background: `${getStatusColor(connectionStatus.anthropic)}20`,
                    color: getStatusColor(connectionStatus.anthropic),
                    borderRadius: 6,
                    fontSize: '0.8rem',
                    fontWeight: 600
                  }}>
                    <span>{getStatusIcon(connectionStatus.anthropic)}</span>
                    {connectionStatus.anthropic === 'connected' ? 'Connected' :
                     connectionStatus.anthropic === 'testing' ? 'Testing...' :
                     connectionStatus.anthropic === 'error' ? 'Connection Failed' :
                     'Not Configured'}
                  </div>
                </div>
                
                {connectionStatus.anthropic === 'connected' && (
                  <div style={{ fontSize: '0.85rem', color: 'var(--muted)' }}>
                    Last tested: 2 minutes ago
                  </div>
                )}
              </div>

              <div style={{ display: 'grid', gap: 16 }}>
                <div>
                  <label style={{ display: 'block', marginBottom: 8, fontWeight: 600, fontSize: '0.9rem' }}>
                    Personal access token
                  </label>
                  <input
                    className="input"
                    type="password"
                    value={apiKeys.anthropic}
                    onChange={(e) => updateApiKey('anthropic', e.target.value)}
                    placeholder="sk-ant-api03-..."
                    style={{ fontFamily: 'ui-monospace, monospace' }}
                  />
                  <div style={{ fontSize: '0.8rem', color: 'var(--muted)', marginTop: 4 }}>
                    Your Anthropic API key. Get one from{' '}
                    <a href="https://console.anthropic.com" target="_blank" rel="noopener noreferrer" style={{ color: 'var(--primary)' }}>
                      console.anthropic.com
                    </a>
                  </div>
                </div>

                <div style={{ display: 'flex', gap: 8 }}>
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
            <div className="panel" style={{ padding: 24, marginBottom: 24 }}>
              <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 20 }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
                  <h4 style={{ margin: 0, fontSize: '1.2rem', fontWeight: 600 }}>
                    OpenAI (GPT)
                  </h4>
                  <div style={{ 
                    display: 'flex', 
                    alignItems: 'center', 
                    gap: 6,
                    padding: '4px 8px',
                    background: `${getStatusColor(connectionStatus.openai)}20`,
                    color: getStatusColor(connectionStatus.openai),
                    borderRadius: 6,
                    fontSize: '0.8rem',
                    fontWeight: 600
                  }}>
                    <span>{getStatusIcon(connectionStatus.openai)}</span>
                    API key required
                  </div>
                </div>
              </div>

              <div style={{ display: 'grid', gap: 16 }}>
                <div>
                  <label style={{ display: 'block', marginBottom: 8, fontWeight: 600, fontSize: '0.9rem' }}>
                    API key
                  </label>
                  <input
                    className="input"
                    type="password"
                    value={apiKeys.openai}
                    onChange={(e) => updateApiKey('openai', e.target.value)}
                    placeholder="sk-..."
                    style={{ fontFamily: 'ui-monospace, monospace' }}
                  />
                  <div style={{ fontSize: '0.8rem', color: 'var(--muted)', marginTop: 4 }}>
                    Your OpenAI API key. Get one from{' '}
                    <a href="https://platform.openai.com/api-keys" target="_blank" rel="noopener noreferrer" style={{ color: 'var(--primary)' }}>
                      platform.openai.com
                    </a>
                  </div>
                </div>

                <div>
                  <label style={{ display: 'block', marginBottom: 8, fontWeight: 600, fontSize: '0.9rem' }}>
                    Organization (optional)
                  </label>
                  <input
                    className="input"
                    placeholder="org-xxxxxxxxxxxxxxxx"
                    style={{ fontFamily: 'ui-monospace, monospace' }}
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
            <div className="panel" style={{ padding: 24, marginBottom: 24 }}>
              <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 20 }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
                  <h4 style={{ margin: 0, fontSize: '1.2rem', fontWeight: 600 }}>
                    Google (Gemini)
                  </h4>
                  <div style={{ 
                    display: 'flex', 
                    alignItems: 'center', 
                    gap: 6,
                    padding: '4px 8px',
                    background: `${getStatusColor(connectionStatus.gemini)}20`,
                    color: getStatusColor(connectionStatus.gemini),
                    borderRadius: 6,
                    fontSize: '0.8rem',
                    fontWeight: 600
                  }}>
                    <span>{getStatusIcon(connectionStatus.gemini)}</span>
                    Testing connection...
                  </div>
                </div>
              </div>

              <div style={{ display: 'grid', gap: 16 }}>
                <div>
                  <label style={{ display: 'block', marginBottom: 8, fontWeight: 600, fontSize: '0.9rem' }}>
                    API key
                  </label>
                  <input
                    className="input"
                    type="password"
                    value={apiKeys.gemini}
                    onChange={(e) => updateApiKey('gemini', e.target.value)}
                    placeholder="AIza..."
                    style={{ fontFamily: 'ui-monospace, monospace' }}
                  />
                </div>

                <div>
                  <label style={{ display: 'block', marginBottom: 8, fontWeight: 600, fontSize: '0.9rem' }}>
                    Project ID
                  </label>
                  <input
                    className="input"
                    placeholder="my-project-123"
                    style={{ fontFamily: 'ui-monospace, monospace' }}
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
            <div className="panel" style={{ padding: 24 }}>
              <h4 style={{ margin: '0 0 16px 0', fontSize: '1.1rem', fontWeight: 600 }}>
                🎯 Global defaults
              </h4>
              
              <div style={{ display: 'grid', gap: 16 }}>
                <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16 }}>
                  <div>
                    <label style={{ display: 'block', marginBottom: 8, fontWeight: 600, fontSize: '0.9rem' }}>
                      Default provider
                    </label>
                    <select className="select">
                      <option value="anthropic">Anthropic (Claude)</option>
                      <option value="openai">OpenAI (GPT)</option>
                      <option value="gemini">Google (Gemini)</option>
                    </select>
                  </div>
                  
                  <div>
                    <label style={{ display: 'block', marginBottom: 8, fontWeight: 600, fontSize: '0.9rem' }}>
                      Fallback provider
                    </label>
                    <select className="select">
                      <option value="anthropic">Anthropic (Claude)</option>
                      <option value="openai">OpenAI (GPT)</option>
                      <option value="gemini">Google (Gemini)</option>
                    </select>
                  </div>
                </div>

                <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16 }}>
                  <div>
                    <label style={{ display: 'block', marginBottom: 8, fontWeight: 600, fontSize: '0.9rem' }}>
                      Global timeout (seconds)
                    </label>
                    <input className="input" type="number" defaultValue="30" />
                  </div>
                  
                  <div>
                    <label style={{ display: 'block', marginBottom: 8, fontWeight: 600, fontSize: '0.9rem' }}>
                      Global max retries
                    </label>
                    <input className="input" type="number" defaultValue="3" />
                  </div>
                </div>
              </div>
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
