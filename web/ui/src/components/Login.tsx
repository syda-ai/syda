import { useState } from 'react'
import { Link } from 'react-router-dom'

type LoginProps = {
  onSubmit?: (data: LoginData) => void
}

type LoginData = {
  username: string
  password: string
}

export default function Login({ onSubmit }: LoginProps) {
  const [formData, setFormData] = useState<LoginData>({
    username: '',
    password: ''
  })
  
  const [errors, setErrors] = useState<Partial<LoginData>>({})
  const [isSubmitting, setIsSubmitting] = useState(false)

  const validateForm = (): boolean => {
    const newErrors: Partial<LoginData> = {}
    
    if (!formData.username.trim()) {
      newErrors.username = 'Username is required'
    }
    
    if (!formData.password) {
      newErrors.password = 'Password is required'
    }
    
    setErrors(newErrors)
    return Object.keys(newErrors).length === 0
  }

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    
    if (!validateForm()) return
    
    setIsSubmitting(true)
    
    try {
      // Simulate API call
      await new Promise(resolve => setTimeout(resolve, 1000))
      onSubmit?.(formData)
    } catch (error) {
      console.error('Login failed:', error)
    } finally {
      setIsSubmitting(false)
    }
  }

  const handleInputChange = (field: keyof LoginData) => (
    e: React.ChangeEvent<HTMLInputElement>
  ) => {
    setFormData(prev => ({ ...prev, [field]: e.target.value }))
    if (errors[field]) {
      setErrors(prev => ({ ...prev, [field]: undefined }))
    }
  }

  return (
    <div className="fade-in" style={{ maxWidth: 400, margin: '0 auto' }}>
      <div className="panel" style={{ padding: 32 }}>
        <div style={{ textAlign: 'center', marginBottom: 32 }}>
          <h2 style={{ 
            margin: 0, 
            marginBottom: 8,
            background: 'linear-gradient(135deg, var(--primary-light), var(--accent-light))', 
            backgroundClip: 'text', 
            WebkitBackgroundClip: 'text', 
            color: 'transparent' 
          }}>
            🔐 Sign In
          </h2>
          <p className="muted" style={{ margin: 0 }}>
            Welcome back to Syda
          </p>
        </div>

        <form onSubmit={handleSubmit} style={{ display: 'grid', gap: 20 }}>
          <div>
            <label style={{ display: 'block', marginBottom: 6, fontWeight: 500 }}>
              Username
            </label>
            <input
              type="text"
              className="input"
              value={formData.username}
              onChange={handleInputChange('username')}
              style={{ width: '100%' }}
              placeholder="Enter your username or email"
              autoComplete="username"
            />
            {errors.username && (
              <div className="danger" style={{ fontSize: '0.85rem', marginTop: 4 }}>
                {errors.username}
              </div>
            )}
          </div>

          <div>
            <label style={{ display: 'block', marginBottom: 6, fontWeight: 500 }}>
              Password
            </label>
            <input
              type="password"
              className="input"
              value={formData.password}
              onChange={handleInputChange('password')}
              style={{ width: '100%' }}
              placeholder="Enter your password"
              autoComplete="current-password"
            />
            {errors.password && (
              <div className="danger" style={{ fontSize: '0.85rem', marginTop: 4 }}>
                {errors.password}
              </div>
            )}
          </div>

          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', fontSize: '0.9rem' }}>
            <label style={{ display: 'flex', alignItems: 'center', gap: 8, cursor: 'pointer' }}>
              <input type="checkbox" style={{ accentColor: 'var(--primary)' }} />
              <span>Remember me</span>
            </label>
            
            <Link
              to="/forgot-password"
              style={{ 
                background: 'none', 
                border: 'none', 
                color: 'var(--primary)', 
                cursor: 'pointer',
                textDecoration: 'underline',
                fontSize: 'inherit'
              }}
            >
              Forgot password?
            </Link>
          </div>

          <button
            type="submit"
            className="btn"
            disabled={isSubmitting}
            style={{ width: '100%', marginTop: 8 }}
          >
            {isSubmitting ? 'Signing In...' : 'Sign In'}
          </button>
        </form>

        <div style={{ textAlign: 'center', marginTop: 24, paddingTop: 24, borderTop: '1px solid var(--border)' }}>
          <p className="muted" style={{ margin: 0, fontSize: '0.85rem' }}>
            Don't have an account? <Link to="/account/create" style={{ color: 'var(--primary)', textDecoration: 'none' }}>Create one</Link>
          </p>
        </div>
      </div>
    </div>
  )
}
