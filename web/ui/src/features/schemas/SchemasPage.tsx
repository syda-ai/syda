import { useMemo, useState } from 'react'
import { useAppState } from '../../store/AppState'
import SpreadsheetSchemaEditor from './SpreadsheetSchemaEditor'
import DatabaseImportModal from './DatabaseImportModal'
import { parseYamlToVisual, fieldsToYaml, databaseTableToYaml, validateSchema } from './schemaUtils'

type EditorMode = 'spreadsheet' | 'yaml'

export default function SchemasPage() {
  const { schemas, setSchemas } = useAppState()
  const [selected, setSelected] = useState<string | null>(schemas[0]?.name || null)
  const [editorMode, setEditorMode] = useState<EditorMode>('spreadsheet')
  const [showImportModal, setShowImportModal] = useState(false)
  const [validationErrors, setValidationErrors] = useState<string[]>([])

  const current = useMemo(() => schemas.find(s => s.name === selected) || null, [schemas, selected])
  
  const parsedSchema = useMemo(() => {
    if (!current || current.kind === 'template') return null
    try {
      return parseYamlToVisual(current.content)
    } catch (error) {
      console.error('Failed to parse YAML:', error)
      return null
    }
  }, [current])

  const updateCurrentSchema = (content: string) => {
    if (!current) return
    setSchemas(prev => prev.map(s => s.name === current.name ? { ...s, content } : s))
    
    // Validate the schema
    if (current.kind === 'structured') {
      try {
        const parsed = parseYamlToVisual(content)
        const errors = validateSchema(parsed)
        setValidationErrors(errors)
      } catch (error) {
        setValidationErrors(['Invalid YAML format'])
      }
    }
  }


  const handleSpreadsheetChange = (fields: any[]) => {
    if (!current) return
    const yamlContent = fieldsToYaml(current.name, `${current.name} schema`, fields)
    updateCurrentSchema(yamlContent)
  }

  const handleDatabaseImport = (tables: any[]) => {
    const newSchemas = tables.map(table => ({
      name: table.name,
      kind: 'structured' as const,
      content: databaseTableToYaml(table)
    }))
    
    setSchemas(prev => [...prev, ...newSchemas])
    setShowImportModal(false)
    
    // Select the first imported schema
    if (newSchemas.length > 0) {
      setSelected(newSchemas[0].name)
    }
  }

  const createNewSchema = () => {
    const name = prompt('New schema name')?.trim()
    if (!name) return
    
    const newSchema = {
      name,
      kind: 'structured' as const,
      content: `__table_name__: ${name}\n__description__: New schema\n\nid:\n  type: integer\n  constraints:\n    primary_key: true\nname:\n  type: string\n`
    }
    
    setSchemas(prev => [...prev, newSchema])
    setSelected(name)
  }

  const deleteCurrentSchema = () => {
    if (!current) return
    if (confirm(`Delete schema "${current.name}"?`)) {
      setSchemas(prev => prev.filter(s => s.name !== current.name))
      setSelected(null)
    }
  }

  return (
    <div className="fade-in" style={{ display: 'grid', gap: 24, width: '100%', maxWidth: '1400px' }}>
      <div style={{ display: 'grid', gridTemplateColumns: '300px 1fr', gap: 20, width: '100%' }}>
        {/* Schema List Sidebar */}
        <div className="panel">
          <div style={{ padding: 16, fontWeight: 700, borderBottom: '1px solid var(--border)', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
            📋 Schemas
            <div style={{ display: 'flex', gap: 4 }}>
              <button 
                className="btn secondary" 
                onClick={() => setShowImportModal(true)}
                style={{ padding: '4px 8px', fontSize: '0.8rem' }}
                title="Import from database"
              >
                🔌
              </button>
              <button 
                className="btn" 
                onClick={createNewSchema}
                style={{ padding: '4px 8px', fontSize: '0.8rem' }}
              >
                + Add
              </button>
            </div>
          </div>
          <div>
            {schemas.map(s => (
              <div key={s.name} onClick={() => setSelected(s.name)} style={{ 
                padding: 12, 
                cursor: 'pointer', 
                background: selected === s.name ? 'linear-gradient(135deg, rgba(59, 130, 246, 0.15), rgba(139, 92, 246, 0.1))' : 'transparent', 
                borderBottom: '1px solid var(--border)',
                transition: 'all 0.2s ease'
              }}>
                <div style={{ fontWeight: 600, display: 'flex', alignItems: 'center', gap: 8 }}>
                  <span>{s.kind === 'template' ? '📄' : '🗂️'}</span>
                  {s.name}
                </div>
                <div className="muted" style={{ fontSize: 12 }}>{s.kind}</div>
              </div>
            ))}
            
            {schemas.length === 0 && (
              <div style={{ padding: 32, textAlign: 'center', color: 'var(--muted)' }}>
                <div style={{ fontSize: '2rem', marginBottom: 8 }}>📋</div>
                <div style={{ fontSize: '0.9rem' }}>No schemas yet</div>
                <button className="btn" onClick={createNewSchema} style={{ marginTop: 12, fontSize: '0.85rem' }}>
                  Create First Schema
                </button>
              </div>
            )}
          </div>
        </div>

        {/* Main Editor Area */}
        <div style={{ display: 'grid', gap: 16 }}>
          {/* Header */}
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
            <div style={{ display: 'flex', gap: 8, alignItems: 'center' }}>
              <h3 style={{ margin: 0 }}>{current?.name || 'No schema selected'}</h3>
              {current && <span className="muted">({current.kind})</span>}
              {validationErrors.length > 0 && (
                <span style={{ 
                  background: 'rgba(239, 68, 68, 0.1)', 
                  color: 'var(--danger)', 
                  padding: '2px 8px', 
                  borderRadius: '4px', 
                  fontSize: '0.8rem' 
                }}>
                  {validationErrors.length} issues
                </span>
              )}
            </div>
            
            <div style={{ display: 'flex', gap: 8 }}>
              {current && current.kind === 'structured' && (
                <div style={{ display: 'flex', background: 'var(--panel)', border: '1px solid var(--border)', borderRadius: '8px', padding: '2px' }}>
                  <button
                    className={editorMode === 'spreadsheet' ? 'btn' : 'btn secondary'}
                    onClick={() => setEditorMode('spreadsheet')}
                    style={{ padding: '6px 12px', fontSize: '0.85rem', margin: 0, borderRadius: '6px' }}
                  >
                    📊 Spreadsheet
                  </button>
                  <button
                    className={editorMode === 'yaml' ? 'btn' : 'btn secondary'}
                    onClick={() => setEditorMode('yaml')}
                    style={{ padding: '6px 12px', fontSize: '0.85rem', margin: 0, borderRadius: '6px' }}
                  >
                    📝 YAML
                  </button>
                </div>
              )}
              
              {current && (
                <button 
                  className="btn secondary"
                  onClick={deleteCurrentSchema}
                  style={{ padding: '6px 12px', fontSize: '0.85rem' }}
                >
                  🗑️ Delete
                </button>
              )}
            </div>
          </div>

          {/* Validation Errors */}
          {validationErrors.length > 0 && (
            <div className="panel" style={{ padding: 12, background: 'rgba(239, 68, 68, 0.1)', border: '1px solid rgba(239, 68, 68, 0.2)' }}>
              <div style={{ fontWeight: 600, marginBottom: 8, color: 'var(--danger)' }}>
                ⚠️ Schema Issues:
              </div>
              <ul style={{ margin: 0, paddingLeft: 20, color: 'var(--danger)', fontSize: '0.9rem' }}>
                {validationErrors.map((error, index) => (
                  <li key={index}>{error}</li>
                ))}
              </ul>
            </div>
          )}

          {/* Editor Content */}
          {!current ? (
            <div className="panel" style={{ padding: 40, textAlign: 'center' }}>
              <div style={{ fontSize: '3rem', marginBottom: 16 }}>📋</div>
              <h3 style={{ margin: '0 0 8px 0' }}>No Schema Selected</h3>
              <p className="muted">Select a schema from the sidebar or create a new one to get started.</p>
              <div style={{ display: 'flex', gap: 8, justifyContent: 'center', marginTop: 16 }}>
                <button className="btn" onClick={createNewSchema}>
                  Create New Schema
                </button>
                <button className="btn secondary" onClick={() => setShowImportModal(true)}>
                  Import from Database
                </button>
              </div>
            </div>
          ) : current.kind === 'template' ? (
            // Template schemas always use YAML editor
            <div>
              <div className="panel" style={{ padding: 12, marginBottom: 16, background: 'rgba(139, 92, 246, 0.1)', border: '1px solid rgba(139, 92, 246, 0.2)' }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: 8, fontSize: '0.9rem' }}>
                  <span>📄</span>
                  <strong>Template Schema</strong> - Templates use YAML configuration for document generation
                </div>
              </div>
              <textarea
                value={current.content}
                onChange={(e) => updateCurrentSchema(e.target.value)}
                className="textarea"
                style={{ 
                  width: '100%', 
                  height: 420, 
                  fontFamily: 'ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace', 
                  fontSize: 13 
                }}
                placeholder="Enter template configuration..."
              />
            </div>
          ) : editorMode === 'spreadsheet' ? (
            // Spreadsheet editor for structured schemas
            <SpreadsheetSchemaEditor
              schemaName={current.name}
              onFieldsChange={handleSpreadsheetChange}
            />
          ) : (
            // YAML editor for structured schemas
            <div>
              <div className="panel" style={{ padding: 12, marginBottom: 16, background: 'rgba(59, 130, 246, 0.1)', border: '1px solid rgba(59, 130, 246, 0.2)' }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: 8, fontSize: '0.9rem' }}>
                  <span>📝</span>
                  <strong>YAML Editor</strong> - Advanced configuration with full control
                </div>
              </div>
              <textarea
                value={current.content}
                onChange={(e) => updateCurrentSchema(e.target.value)}
                className="textarea"
                style={{ 
                  width: '100%', 
                  height: 420, 
                  fontFamily: 'ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace', 
                  fontSize: 13 
                }}
                placeholder="Enter YAML schema definition..."
              />
            </div>
          )}
        </div>
      </div>

      {/* Database Import Modal */}
      {showImportModal && (
        <DatabaseImportModal
          onImport={handleDatabaseImport}
          onClose={() => setShowImportModal(false)}
        />
      )}
    </div>
  )
}



