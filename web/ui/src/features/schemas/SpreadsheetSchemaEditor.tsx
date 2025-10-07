import { useState, useMemo } from 'react'
import { useAppState } from '../../store/AppState'

type FieldRow = {
  id: string
  name: string
  type: string
  primaryKey: boolean
  required: boolean
  foreignKey?: {
    table: string
    column: string
    onDelete?: 'CASCADE' | 'SET NULL' | 'RESTRICT'
  }
  description: string
  constraints?: {
    maxLength?: number
    minValue?: number
    maxValue?: number
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
      primaryKey: true,
      required: true,
      description: 'Primary key'
    }
  ])
  const [editingFK, setEditingFK] = useState<string | null>(null)

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
      primaryKey: false,
      required: false,
      description: ''
    }
    const newFields = [...fields, newField]
    setFields(newFields)
    onFieldsChange(newFields)
  }

  const addMultipleFields = (count: number) => {
    const newFields = Array.from({ length: count }, (_, i) => ({
      id: (Date.now() + i).toString(),
      name: `field_${fields.length + i + 1}`,
      type: 'string',
      primaryKey: false,
      required: false,
      description: ''
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
              <th style={{ padding: '12px 8px', textAlign: 'left', borderRight: '1px solid var(--border)', minWidth: '200px' }}>
                🔗 Foreign Key
              </th>
              <th style={{ padding: '12px 8px', textAlign: 'center', borderRight: '1px solid var(--border)', width: '80px' }}>
                PK
              </th>
              <th style={{ padding: '12px 8px', textAlign: 'center', borderRight: '1px solid var(--border)', width: '80px' }}>
                Required
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
            {fields.map((field, index) => (
              <tr key={field.id} style={{ borderBottom: '1px solid var(--border)' }}>
                {/* Field Name */}
                <td style={{ padding: '8px', borderRight: '1px solid var(--border)' }}>
                  <input
                    className="input"
                    value={field.name}
                    onChange={(e) => updateField(field.id, { name: e.target.value })}
                    style={{ width: '100%', padding: '4px 8px', fontSize: '0.9rem' }}
                  />
                </td>

                {/* Type */}
                <td style={{ padding: '8px', borderRight: '1px solid var(--border)' }}>
                  <select
                    className="select"
                    value={field.type}
                    onChange={(e) => updateField(field.id, { type: e.target.value })}
                    style={{ width: '100%', padding: '4px 8px', fontSize: '0.9rem' }}
                  >
                    <option value="string">String</option>
                    <option value="integer">Integer</option>
                    <option value="float">Float</option>
                    <option value="boolean">Boolean</option>
                    <option value="date">Date</option>
                    <option value="datetime">DateTime</option>
                    <option value="text">Text</option>
                  </select>
                </td>

                {/* Foreign Key */}
                <td style={{ padding: '8px', borderRight: '1px solid var(--border)' }}>
                  <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                    {field.foreignKey ? (
                      <div style={{ 
                        flex: 1, 
                        padding: '4px 8px', 
                        background: 'rgba(59, 130, 246, 0.1)', 
                        border: '1px solid rgba(59, 130, 246, 0.3)',
                        borderRadius: '4px',
                        fontSize: '0.85rem',
                        display: 'flex',
                        alignItems: 'center',
                        gap: 4
                      }}>
                        🔗 {field.foreignKey.table}.{field.foreignKey.column}
                      </div>
                    ) : (
                      <div style={{ flex: 1, color: 'var(--muted)', fontSize: '0.85rem', padding: '4px 8px' }}>
                        No relationship
                      </div>
                    )}
                    <button
                      className="btn secondary"
                      onClick={() => handleFKEdit(field.id)}
                      style={{ padding: '2px 6px', fontSize: '0.8rem' }}
                    >
                      {field.foreignKey ? '✏️' : '🔗'}
                    </button>
                  </div>
                </td>

                {/* Primary Key */}
                <td style={{ padding: '8px', borderRight: '1px solid var(--border)', textAlign: 'center' }}>
                  <input
                    type="checkbox"
                    checked={field.primaryKey}
                    onChange={(e) => updateField(field.id, { primaryKey: e.target.checked })}
                  />
                </td>

                {/* Required */}
                <td style={{ padding: '8px', borderRight: '1px solid var(--border)', textAlign: 'center' }}>
                  <input
                    type="checkbox"
                    checked={field.required}
                    onChange={(e) => updateField(field.id, { required: e.target.checked })}
                  />
                </td>

                {/* Description */}
                <td style={{ padding: '8px', borderRight: '1px solid var(--border)' }}>
                  <input
                    className="input"
                    value={field.description}
                    onChange={(e) => updateField(field.id, { description: e.target.value })}
                    placeholder="Field description..."
                    style={{ width: '100%', padding: '4px 8px', fontSize: '0.9rem' }}
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
    </div>
  )
}
