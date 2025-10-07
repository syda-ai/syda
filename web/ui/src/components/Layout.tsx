import { Link, NavLink } from 'react-router-dom'
import { type ReactNode } from 'react'
import '../styles/theme.css'

type LayoutProps = { children: ReactNode }

export default function Layout({ children }: LayoutProps) {
  return (
    <div className="app-shell-horizontal">
      {/* Top Navigation Bar */}
      <header className="top-nav">
        <div className="brand">
          <Link to="/" style={{ color: 'inherit', textDecoration: 'none', fontWeight: 800, fontSize: '1.1rem' }}>
            Syda UI
          </Link>
        </div>
        <nav className="horizontal-nav">
          <NavLink to="/" end className={({ isActive }) => isActive ? 'active' : ''}>
            🏠 Home
          </NavLink>
          <NavLink to="/schemas" className={({ isActive }) => isActive ? 'active' : ''}>
            📋 Schemas
          </NavLink>
          <NavLink to="/run" className={({ isActive }) => isActive ? 'active' : ''}>
            ⚙️ Run
          </NavLink>
          <NavLink to="/results" className={({ isActive }) => isActive ? 'active' : ''}>
            📊 Results
          </NavLink>
          <NavLink to="/settings" className={({ isActive }) => isActive ? 'active' : ''}>
            ⚙️ Settings
          </NavLink>
        </nav>
        <div className="nav-actions">
          <NavLink to="/account/create" className="btn secondary" style={{ padding: '6px 12px', fontSize: '0.85rem' }}>
            👤 Account
          </NavLink>
        </div>
      </header>
      
      {/* Main Content Area */}
      <main className="main-content">
        {children}
      </main>
    </div>
  )
}



