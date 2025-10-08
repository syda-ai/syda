import { useState, useMemo } from 'react'
import { useAppState } from '../../store/AppState'
import { 
  SYDA_FIELD_TYPES,
  getApplicableConstraints,
  getConstraintInputType,
  getConstraintLabel,
  getConstraintDescription,
  validateConstraintValue
} from './schemaUtils'
import type { SydaFieldType, SydaConstraints } from './schemaUtils'

type FieldRow = {
  id: string
  name: string
  type: SydaFieldType
  description: string
  constraints?: SydaConstraints
  // Legacy support
  primaryKey?: boolean
  required?: boolean
  foreignKey?: {
    table: string
    column: string
    onDelete?: 'CASCADE' | 'SET NULL' | 'RESTRICT'
  }
}

type ForeignKeyModalProps = {
  field: FieldRow
  availableTables: Array<{ name: string; columns: string[] }>
  onSave: (fk: FieldRow['foreignKey']) => void
  onClose: () => void
}

function ForeignKeyModal({ field, availableTables, onSave, onClose }: ForeignKeyModalProps) {
  const [selectedTable, setSelectedTable] = useState(field.foreignKey?.table || '')
  const [selectedColumn, setSelectedColumn] = useState(field.foreignKey?.column || 'id')
  const [onDelete, setOnDelete] = useState(field.foreignKey?.onDelete || 'CASCADE')

  const availableColumns = useMemo(() => {
    const table = availableTables.find(t => t.name === selectedTable)
    return table?.columns || ['id']
  }, [selectedTable, availableTables])

  const handleSave = () => {
    if (!selectedTable) {
      onSave(undefined)
    } else {
      onSave({
        table: selectedTable,
        column: selectedColumn,
        onDelete
      })
    }
    onClose()
  }

  return (
    <div style={{
      position: 'fixed',
      top: 0,
      left: 0,
      right: 0,
      bottom: 0,
      background: 'rgba(0,0,0,0.5)',
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'center',
      zIndex: 1000
    }}>
      <div className="panel" style={{ width: 480, padding: 24 }}>
        <h3 style={{ margin: '0 0 16px 0', display: 'flex', alignItems: 'center', gap: 8 }}>
          🔗 Foreign Key Relationship
        </h3>
        
        <div style={{ marginBottom: 16 }}>
          <div style={{ fontSize: '0.9rem', color: 'var(--muted)', marginBottom: 8 }}>
            Field: <strong>{field.name}</strong> ({field.type})
          </div>
        </div>

        <div style={{ display: 'grid', gap: 16 }}>
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 12 }}>
            <div>
              <label style={{ display: 'block', marginBottom: 4, fontWeight: 600, fontSize: '0.9rem' }}>
                Reference Table
              </label>
              <select
                className="select"
                value={selectedTable}
                onChange={(e) => {
                  setSelectedTable(e.target.value)
                  setSelectedColumn('id') // Reset to default
                }}
              >
                <option value="">-- No Foreign Key --</option>
                {availableTables.map(table => (
                  <option key={table.name} value={table.name}>{table.name}</option>
                ))}
              </select>
            </div>

            <div>
              <label style={{ display: 'block', marginBottom: 4, fontWeight: 600, fontSize: '0.9rem' }}>
                Reference Column
              </label>
              <select
                className="select"
                value={selectedColumn}
                onChange={(e) => setSelectedColumn(e.target.value)}
                disabled={!selectedTable}
              >
                {availableColumns.map(col => (
                  <option key={col} value={col}>{col}</option>
                ))}
              </select>
            </div>
          </div>

          {selectedTable && (
            <div>
              <label style={{ display: 'block', marginBottom: 8, fontWeight: 600, fontSize: '0.9rem' }}>
                On Delete Action
              </label>
              <div style={{ display: 'flex', gap: 16 }}>
                {(['CASCADE', 'SET NULL', 'RESTRICT'] as const).map(action => (
                  <label key={action} style={{ display: 'flex', alignItems: 'center', gap: 4, cursor: 'pointer' }}>
                    <input
                      type="radio"
                      name="onDelete"
                      value={action}
                      checked={onDelete === action}
                      onChange={(e) => setOnDelete(e.target.value as any)}
                    />
                    {action}
                  </label>
                ))}
              </div>
            </div>
          )}

          {selectedTable && (
            <div className="panel" style={{ padding: 12, background: 'rgba(59, 130, 246, 0.1)', border: '1px solid rgba(59, 130, 246, 0.2)' }}>
              <div style={{ fontSize: '0.85rem' }}>
                <strong>Relationship:</strong> {field.name} → {selectedTable}.{selectedColumn}
                <br />
                <strong>Constraint:</strong> ON DELETE {onDelete}
              </div>
            </div>
          )}
        </div>

        <div style={{ display: 'flex', gap: 8, marginTop: 24, justifyContent: 'flex-end' }}>
          <button className="btn secondary" onClick={onClose}>
            Cancel
          </button>
          <button className="btn" onClick={handleSave}>
            {selectedTable ? 'Apply Relationship' : 'Remove Foreign Key'}
          </button>
        </div>
      </div>
    </div>
  )
}

type ConstraintsModalProps = {
  field: FieldRow
  availableSchemas: string[]
  onSave: (constraints: SydaConstraints) => void
  onClose: () => void
}

function ConstraintsModal({ field, availableSchemas, onSave, onClose }: ConstraintsModalProps) {
  const [constraints, setConstraints] = useState<SydaConstraints>(field.constraints || {})
  const [errors, setErrors] = useState<Record<string, string>>({})

  const applicableConstraints = getApplicableConstraints(field.type)

  const updateConstraint = (key: keyof SydaConstraints, value: any) => {
    const newConstraints = { ...constraints }
    
    if (value === '' || value === null || value === undefined) {
      delete newConstraints[key]
    } else {
      newConstraints[key] = value
    }
    
    setConstraints(newConstraints)
    
    // Validate the constraint
    const error = validateConstraintValue(key, value, field.type)
    const newErrors = { ...errors }
    if (error) {
      newErrors[key] = error
    } else {
      delete newErrors[key]
    }
    setErrors(newErrors)
  }

  const handleSave = () => {
    if (Object.keys(errors).length === 0) {
      onSave(constraints)
      onClose()
    }
  }

  const renderConstraintInput = (constraintKey: keyof SydaConstraints) => {
    const inputType = getConstraintInputType(constraintKey)
    const label = getConstraintLabel(constraintKey)
    const description = getConstraintDescription(constraintKey)
    const value = constraints[constraintKey]
    const error = errors[constraintKey]

    switch (inputType) {
      case 'checkbox':
        return (
          <div key={constraintKey} style={{ marginBottom: 16 }}>
            <label style={{ display: 'flex', alignItems: 'center', gap: 8, cursor: 'pointer' }}>
              <input
                type="checkbox"
                checked={!!value}
                onChange={(e) => updateConstraint(constraintKey, e.target.checked)}
              />
              <div>
                <div style={{ fontWeight: 600, fontSize: '0.9rem' }}>{label}</div>
                <div style={{ fontSize: '0.8rem', color: 'var(--muted)' }}>{description}</div>
              </div>
            </label>
          </div>
        )

      case 'number':
        return (
          <div key={constraintKey} style={{ marginBottom: 16 }}>
            <label style={{ display: 'block', marginBottom: 4, fontWeight: 600, fontSize: '0.9rem' }}>
              {label}
            </label>
            <input
              className="input"
              type="number"
              value={typeof value === 'number' ? value : ''}
              onChange={(e) => updateConstraint(constraintKey, e.target.value ? Number(e.target.value) : undefined)}
              placeholder={description}
              style={{ width: '100%' }}
            />
            {error && <div style={{ color: 'var(--danger)', fontSize: '0.8rem', marginTop: 4 }}>{error}</div>}
          </div>
        )

      case 'textarea':
        return (
          <div key={constraintKey} style={{ marginBottom: 16 }}>
            <label style={{ display: 'block', marginBottom: 4, fontWeight: 600, fontSize: '0.9rem' }}>
              {label}
            </label>
            <textarea
              className="textarea"
              value={typeof value === 'string' ? value : ''}
              onChange={(e) => updateConstraint(constraintKey, e.target.value)}
              placeholder={description}
              style={{ width: '100%', height: 80, resize: 'vertical' }}
            />
            {error && <div style={{ color: 'var(--danger)', fontSize: '0.8rem', marginTop: 4 }}>{error}</div>}
            {constraintKey === 'enum' && (
              <div style={{ fontSize: '0.8rem', color: 'var(--muted)', marginTop: 4 }}>
                Example: ["active", "inactive", "pending"] or [1, 2, 3]
              </div>
            )}
          </div>
        )

      case 'select':
        if (constraintKey === 'references') {
          return (
            <div key={constraintKey} style={{ marginBottom: 16 }}>
              <label style={{ display: 'block', marginBottom: 4, fontWeight: 600, fontSize: '0.9rem' }}>
                {label}
              </label>
              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 8 }}>
                <select
                  className="select"
                  value={(value as any)?.schema || ''}
                  onChange={(e) => updateConstraint(constraintKey, e.target.value ? { schema: e.target.value, field: 'id' } : undefined)}
                >
                  <option value="">Select schema</option>
                  {availableSchemas.map(schema => (
                    <option key={schema} value={schema}>{schema}</option>
                  ))}
                </select>
                <input
                  className="input"
                  value={(value as any)?.field || ''}
                  onChange={(e) => updateConstraint(constraintKey, { schema: (value as any)?.schema || '', field: e.target.value })}
                  placeholder="Field name"
                />
              </div>
              <div style={{ fontSize: '0.8rem', color: 'var(--muted)', marginTop: 4 }}>
                {description}
              </div>
            </div>
          )
        }
        break

      default:
        return (
          <div key={constraintKey} style={{ marginBottom: 16 }}>
            <label style={{ display: 'block', marginBottom: 4, fontWeight: 600, fontSize: '0.9rem' }}>
              {label}
            </label>
            <input
              className="input"
              type="text"
              value={typeof value === 'string' ? value : ''}
              onChange={(e) => updateConstraint(constraintKey, e.target.value)}
              placeholder={description}
              style={{ width: '100%' }}
            />
            {error && <div style={{ color: 'var(--danger)', fontSize: '0.8rem', marginTop: 4 }}>{error}</div>}
          </div>
        )
    }
  }

  return (
    <div style={{
      position: 'fixed',
      top: 0,
      left: 0,
      right: 0,
      bottom: 0,
      background: 'rgba(0,0,0,0.5)',
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'center',
      zIndex: 1000
    }}>
      <div className="panel" style={{ width: 600, maxHeight: '80vh', overflow: 'hidden', display: 'flex', flexDirection: 'column' }}>
        <div style={{ padding: 24, borderBottom: '1px solid var(--border)' }}>
          <h3 style={{ margin: '0 0 8px 0', display: 'flex', alignItems: 'center', gap: 8 }}>
            ⚙️ Field Constraints
          </h3>
          <div style={{ fontSize: '0.9rem', color: 'var(--muted)' }}>
            Field: <strong>{field.name}</strong> ({field.type})
          </div>
        </div>

        <div style={{ flex: 1, overflow: 'auto', padding: 24 }}>
          {applicableConstraints.length === 0 ? (
            <div style={{ textAlign: 'center', color: 'var(--muted)', padding: 40 }}>
              No constraints available for this field type.
            </div>
          ) : (
            <div style={{ display: 'grid', gap: 8 }}>
              {applicableConstraints.map(constraintKey => renderConstraintInput(constraintKey))}
            </div>
          )}
        </div>

        <div style={{ padding: 24, borderTop: '1px solid var(--border)', display: 'flex', gap: 8, justifyContent: 'flex-end' }}>
          <button className="btn secondary" onClick={onClose}>
            Cancel
          </button>
          <button 
            className="btn" 
            onClick={handleSave}
            disabled={Object.keys(errors).length > 0}
          >
            Apply Constraints
          </button>
        </div>
      </div>
    </div>
  )
}

type SpreadsheetSchemaEditorProps = {
  schemaName: string
  onFieldsChange: (fields: FieldRow[]) => void
}

export default function SpreadsheetSchemaEditor({ schemaName, onFieldsChange }: SpreadsheetSchemaEditorProps) {
  const { schemas } = useAppState()
  const [fields, setFields] = useState<FieldRow[]>([
    {
      id: '1',
      name: 'id',
      type: 'integer',
      description: 'Primary key',
      constraints: {
        primary_key: true,
        not_null: true
      }
    }
  ])
  const [editingFK, setEditingFK] = useState<string | null>(null)
  const [editingConstraints, setEditingConstraints] = useState<string | null>(null)

  // Mock available tables and their columns (in real app, parse from schemas)
  const availableTables = useMemo(() => {
    return schemas
      .filter(s => s.kind === 'structured' && s.name !== schemaName)
      .map(s => ({
        name: s.name,
        columns: ['id', 'name', 'created_at', 'updated_at'] // In real app, parse from YAML
      }))
  }, [schemas, schemaName])

  const addField = () => {
    const newField: FieldRow = {
      id: Date.now().toString(),
      name: `field_${fields.length + 1}`,
      type: 'string',
      description: '',
      constraints: {}
    }
    const newFields = [...fields, newField]
    setFields(newFields)
    onFieldsChange(newFields)
  }

  const addMultipleFields = (count: number) => {
    const newFields = Array.from({ length: count }, (_, i) => ({
      id: (Date.now() + i).toString(),
      name: `field_${fields.length + i + 1}`,
      type: 'string' as SydaFieldType,
      description: '',
      constraints: {}
    }))
    const updatedFields = [...fields, ...newFields]
    setFields(updatedFields)
    onFieldsChange(updatedFields)
  }

  const updateField = (id: string, updates: Partial<FieldRow>) => {
    const newFields = fields.map(f => f.id === id ? { ...f, ...updates } : f)
    setFields(newFields)
    onFieldsChange(newFields)
  }

  const removeField = (id: string) => {
    const newFields = fields.filter(f => f.id !== id)
    setFields(newFields)
    onFieldsChange(newFields)
  }

  const handleFKEdit = (fieldId: string) => {
    setEditingFK(fieldId)
  }

  const handleFKSave = (fieldId: string, fk: FieldRow['foreignKey']) => {
    updateField(fieldId, { foreignKey: fk })
    setEditingFK(null)
  }

  const handleConstraintsSave = (fieldId: string, constraints: SydaConstraints) => {
    updateField(fieldId, { constraints })
    setEditingConstraints(null)
  }

  // Get available schema names for references
  const availableSchemas = schemas.filter(s => s.kind === 'structured' && s.name !== schemaName).map(s => s.name)

  return (
    <div style={{ display: 'grid', gap: 16 }}>
      {/* Toolbar */}
      <div style={{ display: 'flex', gap: 8, alignItems: 'center' }}>
        <button className="btn" onClick={addField}>
          + Add Field
        </button>
        <button className="btn secondary" onClick={() => addMultipleFields(5)}>
          + Add 5 Fields
        </button>
        <button className="btn secondary" onClick={() => addMultipleFields(10)}>
          + Add 10 Fields
        </button>
        <div style={{ marginLeft: 'auto', fontSize: '0.9rem', color: 'var(--muted)' }}>
          {fields.length} fields
        </div>
      </div>

      {/* Spreadsheet Grid */}
      <div className="panel" style={{ overflow: 'auto', maxHeight: '600px' }}>
        <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '0.9rem' }}>
          <thead>
            <tr style={{ background: 'var(--panel-2)', position: 'sticky', top: 0, zIndex: 10 }}>
              <th style={{ padding: '12px 8px', textAlign: 'left', borderRight: '1px solid var(--border)', minWidth: '150px' }}>
                Field Name
              </th>
              <th style={{ padding: '12px 8px', textAlign: 'left', borderRight: '1px solid var(--border)', minWidth: '120px' }}>
                Type
              </th>
              <th style={{ padding: '12px 8px', textAlign: 'left', borderRight: '1px solid var(--border)', minWidth: '250px' }}>
                ⚙️ Constraints
              </th>
              <th style={{ padding: '12px 8px', textAlign: 'left', borderRight: '1px solid var(--border)', minWidth: '200px' }}>
                Description
              </th>
              <th style={{ padding: '12px 8px', textAlign: 'center', width: '60px' }}>
                Actions
              </th>
            </tr>
          </thead>
          <tbody>
            {fields.map((field) => (
              <tr key={field.id} style={{ borderBottom: '1px solid var(--border)' }}>
                {/* Field Name */}
                <td style={{ padding: '8px', borderRight: '1px solid var(--border)' }}>
                  <input
                    className="input"
                    value={field.name}
                    onChange={(e) => updateField(field.id, { name: e.target.value })}
                    style={{ width: '100%', padding: '4px 8px', fontSize: '0.9rem', boxSizing: 'border-box' }}
                  />
                </td>

                {/* Type */}
                <td style={{ padding: '8px', borderRight: '1px solid var(--border)' }}>
                  <select
                    className="select"
                    value={field.type}
                    onChange={(e) => updateField(field.id, { type: e.target.value as SydaFieldType })}
                    style={{ width: '100%', padding: '4px 8px', fontSize: '0.9rem', boxSizing: 'border-box' }}
                  >
                    <optgroup label="Core Types">
                      {SYDA_FIELD_TYPES.filter(t => t.category === 'core').map(type => (
                        <option key={type.value} value={type.value}>{type.label}</option>
                      ))}
                    </optgroup>
                    <optgroup label="Extended Types">
                      {SYDA_FIELD_TYPES.filter(t => t.category === 'extended').map(type => (
                        <option key={type.value} value={type.value}>{type.label}</option>
                      ))}
                    </optgroup>
                  </select>
                </td>

                {/* Constraints */}
                <td style={{ padding: '8px', borderRight: '1px solid var(--border)' }}>
                  <div style={{ display: 'flex', alignItems: 'center', gap: 4, flexWrap: 'wrap' }}>
                    {/* Show active constraints as badges */}
                    {field.constraints?.primary_key && (
                      <span style={{ 
                        background: 'rgba(255, 193, 7, 0.2)', 
                        color: '#f59e0b', 
                        padding: '2px 6px', 
                        borderRadius: '4px', 
                        fontSize: '0.7rem',
                        fontWeight: 600
                      }}>
                        PK
                      </span>
                    )}
                    {field.constraints?.unique && (
                      <span style={{ 
                        background: 'rgba(139, 92, 246, 0.2)', 
                        color: '#8b5cf6', 
                        padding: '2px 6px', 
                        borderRadius: '4px', 
                        fontSize: '0.7rem',
                        fontWeight: 600
                      }}>
                        UNIQUE
                      </span>
                    )}
                    {field.constraints?.not_null && (
                      <span style={{ 
                        background: 'rgba(239, 68, 68, 0.2)', 
                        color: '#ef4444', 
                        padding: '2px 6px', 
                        borderRadius: '4px', 
                        fontSize: '0.7rem',
                        fontWeight: 600
                      }}>
                        NOT NULL
                      </span>
                    )}
                    {field.constraints?.enum && (
                      <span style={{ 
                        background: 'rgba(34, 197, 94, 0.2)', 
                        color: '#22c55e', 
                        padding: '2px 6px', 
                        borderRadius: '4px', 
                        fontSize: '0.7rem',
                        fontWeight: 600
                      }}>
                        ENUM
                      </span>
                    )}
                    {field.constraints?.references && (
                      <span style={{ 
                        background: 'rgba(59, 130, 246, 0.2)', 
                        color: '#3b82f6', 
                        padding: '2px 6px', 
                        borderRadius: '4px', 
                        fontSize: '0.7rem',
                        fontWeight: 600
                      }}>
                        FK → {field.constraints.references.schema}
                      </span>
                    )}
                    {(field.constraints?.min !== undefined || field.constraints?.max !== undefined) && (
                      <span style={{ 
                        background: 'rgba(168, 85, 247, 0.2)', 
                        color: '#a855f7', 
                        padding: '2px 6px', 
                        borderRadius: '4px', 
                        fontSize: '0.7rem',
                        fontWeight: 600
                      }}>
                        RANGE
                      </span>
                    )}
                    
                    {/* Constraints button */}
                    <button
                      className="btn secondary"
                      onClick={() => setEditingConstraints(field.id)}
                      style={{ padding: '2px 6px', fontSize: '0.8rem', marginLeft: 'auto' }}
                      title="Edit constraints"
                    >
                      ⚙️
                    </button>
                  </div>
                </td>

                {/* Description */}
                <td style={{ padding: '8px', borderRight: '1px solid var(--border)' }}>
                  <input
                    className="input"
                    value={field.description}
                    onChange={(e) => updateField(field.id, { description: e.target.value })}
                    placeholder="Field description..."
                    style={{ width: '100%', padding: '4px 8px', fontSize: '0.9rem', boxSizing: 'border-box' }}
                  />
                </td>

                {/* Actions */}
                <td style={{ padding: '8px', textAlign: 'center' }}>
                  <button
                    className="btn secondary"
                    onClick={() => removeField(field.id)}
                    style={{ padding: '2px 6px', fontSize: '0.8rem' }}
                    title="Delete field"
                  >
                    🗑️
                  </button>
                </td>
              </tr>
            ))}
          </tbody>
        </table>

        {fields.length === 0 && (
          <div style={{ padding: 40, textAlign: 'center', color: 'var(--muted)' }}>
            No fields defined. Click "Add Field" to get started.
          </div>
        )}
      </div>

      {/* Foreign Key Modal */}
      {editingFK && (
        <ForeignKeyModal
          field={fields.find(f => f.id === editingFK)!}
          availableTables={availableTables}
          onSave={(fk) => handleFKSave(editingFK, fk)}
          onClose={() => setEditingFK(null)}
        />
      )}

      {/* Constraints Modal */}
      {editingConstraints && (
        <ConstraintsModal
          field={fields.find(f => f.id === editingConstraints)!}
          availableSchemas={availableSchemas}
          onSave={(constraints) => handleConstraintsSave(editingConstraints, constraints)}
          onClose={() => setEditingConstraints(null)}
        />
      )}
    </div>
  )
}
