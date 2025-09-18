import { useTheme } from '../../store/ThemeContext'

export default function SettingsPage() {
  const { theme, setTheme } = useTheme()

  return (
    <div className="fade-in" style={{ display: 'grid', gap: 24, maxWidth: 600 }}>
      <div>
        <h2 style={{ 
          margin: 0, 
          marginBottom: 8,
          background: 'linear-gradient(135deg, var(--primary-light), var(--accent-light))', 
          backgroundClip: 'text', 
          WebkitBackgroundClip: 'text', 
          color: 'transparent' 
        }}>
          ⚙️ Settings
        </h2>
        <p className="muted" style={{ margin: 0 }}>Customize your Syda UI experience</p>
      </div>

      <div className="panel" style={{ padding: 20 }}>
        <div style={{ display: 'grid', gap: 16 }}>
          <div>
            <h3 style={{ margin: 0, marginBottom: 8, display: 'flex', alignItems: 'center', gap: 8 }}>
              🎨 Appearance
            </h3>
            <p className="muted" style={{ margin: 0, fontSize: '0.9rem' }}>
              Choose between light and dark themes
            </p>
          </div>

          <div style={{ display: 'grid', gap: 12 }}>
            <label style={{ display: 'flex', alignItems: 'center', gap: 12, cursor: 'pointer' }}>
              <input 
                type="radio" 
                name="theme" 
                value="dark" 
                checked={theme === 'dark'} 
                onChange={() => setTheme('dark')}
                style={{ accentColor: 'var(--primary)' }}
              />
              <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                <span style={{ fontSize: '1.2rem' }}>🌙</span>
                <div>
                  <div style={{ fontWeight: 600 }}>Dark Theme</div>
                  <div className="muted" style={{ fontSize: '0.85rem' }}>Easy on the eyes in low light</div>
                </div>
              </div>
            </label>

            <label style={{ display: 'flex', alignItems: 'center', gap: 12, cursor: 'pointer' }}>
              <input 
                type="radio" 
                name="theme" 
                value="light" 
                checked={theme === 'light'} 
                onChange={() => setTheme('light')}
                style={{ accentColor: 'var(--primary)' }}
              />
              <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                <span style={{ fontSize: '1.2rem' }}>☀️</span>
                <div>
                  <div style={{ fontWeight: 600 }}>Light Theme</div>
                  <div className="muted" style={{ fontSize: '0.85rem' }}>Clean and bright interface</div>
                </div>
              </div>
            </label>
          </div>
        </div>
      </div>

      <div className="panel" style={{ padding: 20 }}>
        <div style={{ display: 'grid', gap: 16 }}>
          <div>
            <h3 style={{ margin: 0, marginBottom: 8, display: 'flex', alignItems: 'center', gap: 8 }}>
              ℹ️ About
            </h3>
          </div>
          
          <div style={{ display: 'grid', gap: 8, fontSize: '0.9rem' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between' }}>
              <span className="muted">Version</span>
              <span>1.0.0</span>
            </div>
            <div style={{ display: 'flex', justifyContent: 'space-between' }}>
              <span className="muted">Build</span>
              <span>dev</span>
            </div>
            <div style={{ display: 'flex', justifyContent: 'space-between' }}>
              <span className="muted">Framework</span>
              <span>React + TypeScript</span>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}
