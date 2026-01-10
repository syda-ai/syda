import { BrowserRouter, Route, Routes } from 'react-router-dom'
import Layout from './components/Layout'
import DependencyGraph from './features/deps/DependencyGraph'
import SchemasPage from './features/schemas/SchemasPage'
import RunPage from './features/run/RunPage'
import ResultsPage from './features/results/ResultsPage'
import SettingsPage from './features/settings/SettingsPage'
import AccountCreation from './components/AccountCreation'
import Login from './components/Login'
import ForgotPassword from './components/ForgotPassword'
import './App.css'
import { AppStateProvider } from './store/AppState'
import { ThemeProvider } from './store/ThemeContext'
import { ModalProvider } from './components/ModalProvider'

function Home() {
  return (
    <div className="fade-in page-container centered-800">
      <div className="page-section-header">
        <h1 className="m-0 mb-16 font-800" style={{ fontSize: '3rem' }}>
          Welcome to Syda UI
        </h1>
        <p className="text-lg" style={{ color: 'var(--muted)', maxWidth: 600, margin: '0 auto' }}>
          AI-powered synthetic data generation with referential integrity. 
          Configure models, manage schemas, and generate realistic test data.
        </p>
      </div>
      
      <div className="grid grid-cards gap-20">
        <div className="panel p-24 text-center">
          <div style={{ fontSize: '2rem', marginBottom: 12 }}>🧩</div>
          <h3 className="m-0 mb-8">Schemas</h3>
          <p className="muted m-0 text-xs">Define and edit data schemas with YAML, JSON, or SQLAlchemy models</p>
        </div>
        
        <div className="panel p-24 text-center">
          <div style={{ fontSize: '2rem', marginBottom: 12 }}>⚙️</div>
          <h3 className="m-0 mb-8">Generate</h3>
          <p className="muted m-0 text-xs">Run AI-powered generation with custom prompts and sample sizes</p>
        </div>
        
        <div className="panel p-24 text-center">
          <div style={{ fontSize: '2rem', marginBottom: 12 }}>📊</div>
          <h3 className="m-0 mb-8">Results</h3>
          <p className="muted m-0 text-xs">Browse generated data and download as CSV or JSON</p>
        </div>
      </div>
    </div>
  )
}

export default function App() {
  return (
    <ThemeProvider>
      <AppStateProvider>
        <ModalProvider>
          <BrowserRouter>
            <Layout>
              <Routes>
                <Route path="/" element={<Home />} />
                <Route path="/dependencies" element={<DependencyGraph />} />
                <Route path="/schemas" element={<SchemasPage />} />
                <Route path="/run" element={<RunPage />} />
                <Route path="/results" element={<ResultsPage />} />
                <Route path="/settings" element={<SettingsPage />} />
                <Route path="/account/create" element={<AccountCreation />} />
                <Route path="/login" element={<Login />} />
                <Route path="/forgot-password" element={<ForgotPassword />} />
              </Routes>
            </Layout>
          </BrowserRouter>
        </ModalProvider>
      </AppStateProvider>
    </ThemeProvider>
  )
}
