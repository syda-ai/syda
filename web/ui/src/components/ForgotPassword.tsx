import { useState } from 'react'
import { Link } from 'react-router-dom'

type ForgotPasswordProps = {
  onSubmit?: (data: ForgotPasswordData) => void
}

type ForgotPasswordData = {
  username: string
}

export default function ForgotPassword({ onSubmit }: ForgotPasswordProps) {
  const [formData, setFormData] = useState<ForgotPasswordData>({
    username: ''
  })
  
  const [errors, setErrors] = useState<Partial<ForgotPasswordData>>({})
  const [isSubmitting, setIsSubmitting] = useState(false)
  const [isSubmitted, setIsSubmitted] = useState(false)

  const validateForm = (): boolean => {
    const newErrors: Partial<ForgotPasswordData> = {}
    
    if (!formData.username.trim()) {
      newErrors.username = 'Username is required'
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
      await new Promise(resolve => setTimeout(resolve, 1500))
      onSubmit?.(formData)
      setIsSubmitted(true)
    } catch (error) {
      console.error('Password reset request failed:', error)
    } finally {
      setIsSubmitting(false)
    }
  }

  const handleInputChange = (field: keyof ForgotPasswordData) => (
    e: React.ChangeEvent<HTMLInputElement>
  ) => {
    setFormData(prev => ({ ...prev, [field]: e.target.value }))
    if (errors[field]) {
      setErrors(prev => ({ ...prev, [field]: undefined }))
    }
  }

  if (isSubmitted) {
    return (
      <div className="fade-in" style={{ display: 'grid', gap: 24, width: 800, maxWidth: 800 }}>
    <div style={{ width: 480, maxWidth: 480, margin: '0 auto' }}>
        <div className="panel" style={{ padding: 32, textAlign: 'center' }}>
          <div style={{ fontSize: '3rem', marginBottom: 16 }}>📧</div>
          <h2 style={{ 
            margin: 0, 
            marginBottom: 16,
            background: 'linear-gradient(135deg, var(--success), var(--primary-light))', 
            backgroundClip: 'text', 
            WebkitBackgroundClip: 'text', 
            color: 'transparent' 
          }}>
            Check Your Email
          </h2>
          <p className="muted" style={{ margin: 0, marginBottom: 24, lineHeight: 1.5 }}>
            We've sent password reset instructions to the email associated with your username.
          </p>
          <p className="muted" style={{ margin: 0, marginBottom: 24, fontSize: '0.9rem' }}>
            Didn't receive the email? Check your spam folder or try again.
          </p>
          
          <div style={{ display: 'flex', gap: 12 }}>
            <Link to="/login" className="btn" style={{ flex: 1, textDecoration: 'none', textAlign: 'center' }}>
              Back to Sign In
            </Link>
            <button 
              className="btn secondary" 
              onClick={() => {
                setIsSubmitted(false)
                setFormData({ username: '' })
              }}
              style={{ flex: 1 }}
            >
              Try Again
            </button>
          </div>
        </div>
      </div>
      </div>
    )
  }

  return (
    <div className="fade-in" style={{ display: 'grid', gap: 24, width: 800, maxWidth: 800 }}>
    <div style={{ width: 480, maxWidth: 480, margin: '0 auto' }}>
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
            🔑 Forgot Password
          </h2>
          <p className="muted" style={{ margin: 0, lineHeight: 1.5 }}>
            Enter your username and we'll send you instructions to reset your password
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
              disabled={isSubmitting}
            />
            {errors.username && (
              <div className="danger" style={{ fontSize: '0.85rem', marginTop: 4 }}>
                {errors.username}
              </div>
            )}
          </div>

          <button
            type="submit"
            className="btn"
            disabled={isSubmitting}
            style={{ width: '100%', marginTop: 8 }}
          >
            {isSubmitting ? 'Sending Instructions...' : 'Send Reset Instructions'}
          </button>
        </form>

        <div style={{ textAlign: 'center', marginTop: 24, paddingTop: 24, borderTop: '1px solid var(--border)' }}>
          <p className="muted" style={{ margin: 0, fontSize: '0.85rem' }}>
            Remember your password? <Link to="/login" style={{ color: 'var(--primary)', textDecoration: 'none' }}>Back to Sign In</Link>
          </p>
        </div>
      </div>
    </div>
    </div>
  )
}
