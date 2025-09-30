import { Link, NavLink } from 'react-router-dom'
import { type ReactNode } from 'react'
import '../styles/theme.css'

type LayoutProps = { children: ReactNode }

export default function Layout({ children }: LayoutProps) {
  return (
    <div className="app-shell">
      <aside className="sidebar">
        <div className="brand"><Link to="/" style={{ color: 'inherit', textDecoration: 'none' }}>Syda UI</Link></div>
        <nav className="nav">
          <NavLink to="/" end className={({ isActive }) => isActive ? 'active' : ''}>🏠 Home</NavLink>
          <NavLink to="/schemas" className={({ isActive }) => isActive ? 'active' : ''}>🧩 Schemas</NavLink>
          <NavLink to="/run" className={({ isActive }) => isActive ? 'active' : ''}>⚙️ Run</NavLink>
          <NavLink to="/results" className={({ isActive }) => isActive ? 'active' : ''}>📊 Results</NavLink>
          <NavLink to="/account/create" className={({ isActive }) => isActive ? 'active' : ''}>👤 Account</NavLink>
          <NavLink to="/settings" className={({ isActive }) => isActive ? 'active' : ''}>⚙️ Settings</NavLink>
        </nav>
      </aside>
      <section className="content">
        <div className="toolbar" />
        <div className="main">{children}</div>
      </section>
    </div>
  )
}



