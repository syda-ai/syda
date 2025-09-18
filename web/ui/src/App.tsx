import { BrowserRouter, Route, Routes } from 'react-router-dom'
import Layout from './components/Layout'
import DependencyGraph from './features/deps/DependencyGraph'
import SchemasPage from './features/schemas/SchemasPage'
import RunPage from './features/run/RunPage'
import ResultsPage from './features/results/ResultsPage'
import SettingsPage from './features/settings/SettingsPage'
import './App.css'
import { AppStateProvider } from './store/AppState'
import { ThemeProvider } from './store/ThemeContext'

function Home() {
  return (
    <div className="fade-in" style={{ display: 'grid', gap: 24, maxWidth: 800 }}>
      <div style={{ textAlign: 'center', padding: '40px 0' }}>
        <h1 style={{ 
          margin: 0, 
          fontSize: '3rem', 
          fontWeight: 800,
          background: 'linear-gradient(135deg, var(--primary-light), var(--accent-light))', 
          backgroundClip: 'text', 
          WebkitBackgroundClip: 'text', 
          color: 'transparent',
          marginBottom: 16
        }}>
          Welcome to Syda UI
        </h1>
        <p style={{ fontSize: '1.25rem', color: 'var(--muted)', maxWidth: 600, margin: '0 auto' }}>
          AI-powered synthetic data generation with referential integrity. 
          Configure models, manage schemas, and generate realistic test data.
        </p>
      </div>
      
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))', gap: 20 }}>
        <div className="panel" style={{ padding: 24, textAlign: 'center' }}>
          <div style={{ fontSize: '2rem', marginBottom: 12 }}>🧩</div>
          <h3 style={{ margin: 0, marginBottom: 8 }}>Schemas</h3>
          <p className="muted" style={{ margin: 0, fontSize: '0.9rem' }}>Define and edit data schemas with YAML, JSON, or SQLAlchemy models</p>
        </div>
        
        <div className="panel" style={{ padding: 24, textAlign: 'center' }}>
          <div style={{ fontSize: '2rem', marginBottom: 12 }}>⚙️</div>
          <h3 style={{ margin: 0, marginBottom: 8 }}>Generate</h3>
          <p className="muted" style={{ margin: 0, fontSize: '0.9rem' }}>Run AI-powered generation with custom prompts and sample sizes</p>
        </div>
        
        <div className="panel" style={{ padding: 24, textAlign: 'center' }}>
          <div style={{ fontSize: '2rem', marginBottom: 12 }}>📊</div>
          <h3 style={{ margin: 0, marginBottom: 8 }}>Results</h3>
          <p className="muted" style={{ margin: 0, fontSize: '0.9rem' }}>Browse generated data and download as CSV or JSON</p>
        </div>
      </div>
    </div>
  )
}

export default function App() {
  return (
    <ThemeProvider>
      <AppStateProvider>
        <BrowserRouter>
          <Layout>
            <Routes>
              <Route path="/" element={<Home />} />
              <Route path="/dependencies" element={<DependencyGraph />} />
              <Route path="/schemas" element={<SchemasPage />} />
              <Route path="/run" element={<RunPage />} />
              <Route path="/results" element={<ResultsPage />} />
              <Route path="/settings" element={<SettingsPage />} />
            </Routes>
          </Layout>
        </BrowserRouter>
      </AppStateProvider>
    </ThemeProvider>
  )
}
