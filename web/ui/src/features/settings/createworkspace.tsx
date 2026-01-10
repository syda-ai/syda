import { useState, useEffect } from 'react'
import { apiJson } from '../../lib/apiClient'
import { useModal } from '../../components/ModalProvider'

type CreateWorkspaceProps = {
  onClose?: () => void
  onCreate?: (data: { name: string, description: string }) => void
  onUpdate?: (data: { name: string, description: string }) => void
  mode?: 'create' | 'edit'
  initial?: { name?: string, description?: string }
  workspaceId?: string
}

export default function CreateWorkspace({ onClose, onCreate, onUpdate, mode = 'create', initial, workspaceId }: CreateWorkspaceProps) {
  const [name, setName] = useState<string>(initial?.name || '')
  const [description, setDescription] = useState<string>(initial?.description || '')
  const [isSubmitting, setIsSubmitting] = useState<boolean>(false)
  const { alert } = useModal()

  useEffect(() => {
    // Keep inputs in sync if initial changes (e.g., switching edit targets)
    if (initial) {
      setName(initial.name || '')
      setDescription(initial.description || '')
    }
  }, [initial?.name, initial?.description])

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    if (!name.trim()) {
      await alert({
        title: 'Validation',
        message: 'Workspace name is required'
      })
      return
    }
    const payload = { name: name.trim(), description: description.trim() }
    setIsSubmitting(true)
    try {
      if (mode === 'edit') {
        if (!workspaceId) {
          throw new Error('Missing workspace id for update')
        }
        await apiJson(`/workspaces/${workspaceId}`, payload, { method: 'PUT' })
        await alert({
          title: 'Update Workspace',
          message: 'Workspace updated successfully'
        })
        if (onUpdate) onUpdate(payload)
      } else {
        await apiJson('/workspaces/', payload, { method: 'POST' })
        await alert({
          title: 'Create Workspace',
          message: 'Workspace created successfully'
        })
        if (onCreate) onCreate(payload)
      }
      if (onClose) onClose()
    } catch (err: unknown) {
      const message = err instanceof Error ? err.message : String(err)
      await alert({
        title: `${mode === 'edit' ? 'Update' : 'Create'} Workspace Failed`,
        message: `Failed to ${mode === 'edit' ? 'update' : 'create'} workspace: ${message}`
      })
    } finally {
      setIsSubmitting(false)
    }
  }

  return (
    <div className="panel" style={{ padding: 24 }}>
      <h4 style={{ margin: '0 0 16px 0', fontSize: '1.1rem', fontWeight: 600 }}>
        {mode === 'edit' ? 'Edit workspace' : 'Create workspace'}
      </h4>
      <form onSubmit={handleSubmit} style={{ display: 'grid', gap: 16 }}>
        <div>
          <label style={{ display: 'block', marginBottom: 8, fontWeight: 600, fontSize: '0.9rem' }}>
            Name
          </label>
          <input
            className="input"
            type="text"
            placeholder="Enter workspace name"
            value={name}
            onChange={(e) => setName(e.target.value)}
            disabled={isSubmitting}
          />
        </div>
        <div>
          <label style={{ display: 'block', marginBottom: 8, fontWeight: 600, fontSize: '0.9rem' }}>
            Description
          </label>
          <textarea
            className="input"
            placeholder="Describe this workspace (optional)"
            value={description}
            onChange={(e) => setDescription(e.target.value)}
            rows={5}
            disabled={isSubmitting}
          />
        </div>
        <div style={{ display: 'flex', gap: 8 }}>
          <button className="btn" type="submit" disabled={isSubmitting}>
            {isSubmitting ? (mode === 'edit' ? '⏳ Updating...' : '⏳ Creating...') : (mode === 'edit' ? '✅ Update' : '✅ Create')}
          </button>
          {onClose && (
            <button className="btn secondary" type="button" onClick={onClose} disabled={isSubmitting}>
              ✖ Cancel
            </button>
          )}
        </div>
      </form>
    </div>
  )
}

