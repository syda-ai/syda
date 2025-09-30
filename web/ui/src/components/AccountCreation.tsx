import { useState } from 'react'
import { Link } from 'react-router-dom'

type AccountCreationProps = {
  onSubmit?: (data: AccountData) => void
  onCancel?: () => void
}

type AccountData = {
  firstName: string
  lastName: string
  email: string
  phoneNumber: string
}

export default function AccountCreation({ onSubmit, onCancel }: AccountCreationProps) {
  const [formData, setFormData] = useState<AccountData>({
    firstName: '',
    lastName: '',
    email: '',
    phoneNumber: ''
  })
  
  const [errors, setErrors] = useState<Partial<AccountData>>({})
  const [isSubmitting, setIsSubmitting] = useState(false)

  const validateForm = (): boolean => {
    const newErrors: Partial<AccountData> = {}
    
    if (!formData.firstName.trim()) {
      newErrors.firstName = 'First name is required'
    }
    
    if (!formData.lastName.trim()) {
      newErrors.lastName = 'Last name is required'
    }
    
    if (!formData.email.trim()) {
      newErrors.email = 'Email is required'
    } else if (!/^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(formData.email)) {
      newErrors.email = 'Please enter a valid email address'
    }
    
    if (!formData.phoneNumber.trim()) {
      newErrors.phoneNumber = 'Phone number is required'
    } else if (!/^[\+]?[1-9][\d]{0,15}$/.test(formData.phoneNumber.replace(/[\s\-\(\)]/g, ''))) {
      newErrors.phoneNumber = 'Please enter a valid phone number'
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
      console.error('Account creation failed:', error)
    } finally {
      setIsSubmitting(false)
    }
  }

  const handleInputChange = (field: keyof AccountData) => (
    e: React.ChangeEvent<HTMLInputElement>
  ) => {
    setFormData(prev => ({ ...prev, [field]: e.target.value }))
    if (errors[field]) {
      setErrors(prev => ({ ...prev, [field]: undefined }))
    }
  }

  return (
    <div className="fade-in" style={{ maxWidth: 480, margin: '0 auto' }}>
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
            👤 Create Account
          </h2>
          <p className="muted" style={{ margin: 0 }}>
            Enter your details to get started
          </p>
        </div>

        <form onSubmit={handleSubmit} style={{ display: 'grid', gap: 20 }}>
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16 }}>
            <div>
              <label style={{ display: 'block', marginBottom: 6, fontWeight: 500 }}>
                First Name
              </label>
              <input
                type="text"
                className="input"
                value={formData.firstName}
                onChange={handleInputChange('firstName')}
                style={{ width: '100%' }}
                placeholder="Enter your first name"
              />
              {errors.firstName && (
                <div className="danger" style={{ fontSize: '0.85rem', marginTop: 4 }}>
                  {errors.firstName}
                </div>
              )}
            </div>
            
            <div>
              <label style={{ display: 'block', marginBottom: 6, fontWeight: 500 }}>
                Last Name
              </label>
              <input
                type="text"
                className="input"
                value={formData.lastName}
                onChange={handleInputChange('lastName')}
                style={{ width: '100%' }}
                placeholder="Enter your last name"
              />
              {errors.lastName && (
                <div className="danger" style={{ fontSize: '0.85rem', marginTop: 4 }}>
                  {errors.lastName}
                </div>
              )}
            </div>
          </div>

          <div>
            <label style={{ display: 'block', marginBottom: 6, fontWeight: 500 }}>
              Email ID
            </label>
            <input
              type="email"
              className="input"
              value={formData.email}
              onChange={handleInputChange('email')}
              style={{ width: '100%' }}
              placeholder="Enter your email address"
            />
            {errors.email && (
              <div className="danger" style={{ fontSize: '0.85rem', marginTop: 4 }}>
                {errors.email}
              </div>
            )}
          </div>

          <div>
            <label style={{ display: 'block', marginBottom: 6, fontWeight: 500 }}>
              Phone Number
            </label>
            <input
              type="tel"
              className="input"
              value={formData.phoneNumber}
              onChange={handleInputChange('phoneNumber')}
              style={{ width: '100%' }}
              placeholder="Enter your phone number"
            />
            {errors.phoneNumber && (
              <div className="danger" style={{ fontSize: '0.85rem', marginTop: 4 }}>
                {errors.phoneNumber}
              </div>
            )}
          </div>

          <div style={{ display: 'flex', gap: 12, marginTop: 8 }}>
            <button
              type="submit"
              className="btn"
              disabled={isSubmitting}
              style={{ flex: 1 }}
            >
              {isSubmitting ? 'Creating Account...' : 'Create Account'}
            </button>
            
            {onCancel && (
              <button
                type="button"
                className="btn secondary"
                onClick={onCancel}
                disabled={isSubmitting}
              >
                Cancel
              </button>
            )}
          </div>
        </form>

        <div style={{ textAlign: 'center', marginTop: 24, paddingTop: 24, borderTop: '1px solid var(--border)' }}>
          <p className="muted" style={{ margin: 0, fontSize: '0.85rem' }}>
            Already have an account? <Link to="/login" style={{ color: 'var(--primary)', textDecoration: 'none' }}>Sign in</Link>
          </p>
        </div>
      </div>
    </div>
  )
}
