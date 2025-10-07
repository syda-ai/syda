import { useMemo, useState } from 'react'
import { useAppState } from '../../store/AppState'
import SpreadsheetSchemaEditor from './SpreadsheetSchemaEditor'
import DatabaseImportModal from './DatabaseImportModal'
import TreeView from './TreeView'
import { parseYamlToVisual, fieldsToYaml, databaseTableToYaml, validateSchema } from './schemaUtils'

type EditorMode = 'spreadsheet' | 'yaml'

export default function SchemasPage() {
  const { schemaGroups, setSchemaGroups, selectedSchema, setSelectedSchema } = useAppState()
  const [editorMode, setEditorMode] = useState<EditorMode>('spreadsheet')
  const [showImportModal, setShowImportModal] = useState(false)
  const [validationErrors, setValidationErrors] = useState<string[]>([])

  // Find current schema across all groups
  const currentSchema = useMemo(() => {
    for (const group of schemaGroups) {
      const schema = group.schemas.find(s => s.id === selectedSchema)
      if (schema) return schema
    }
    return null
  }, [schemaGroups, selectedSchema])

  // Find current group
  const currentGroup = useMemo(() => {
    return schemaGroups.find(group => 
      group.schemas.some(s => s.id === selectedSchema)
    )
  }, [schemaGroups, selectedSchema])

  const updateCurrentSchema = (content: string) => {
    if (!currentSchema) return
    
    setSchemaGroups(prev => prev.map(group => ({
      ...group,
      schemas: group.schemas.map(schema => 
        schema.id === currentSchema.id 
          ? { ...schema, content, metadata: { ...schema.metadata, lastModified: new Date() } }
          : schema
      )
    })))
    
    // Validate the schema
    if (currentSchema.kind === 'structured') {
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
    if (!currentSchema) return
    const yamlContent = fieldsToYaml(currentSchema.name, `${currentSchema.name} schema`, fields)
    updateCurrentSchema(yamlContent)
  }

  const handleToggleGroup = (groupId: string) => {
    setSchemaGroups(prev => prev.map(group => 
      group.id === groupId ? { ...group, expanded: !group.expanded } : group
    ))
  }

  const handleSchemaAction = (action: string, schemaId: string) => {
    switch (action) {
      case 'spreadsheet':
        setSelectedSchema(schemaId)
        setEditorMode('spreadsheet')
        break
      case 'yaml':
        setSelectedSchema(schemaId)
        setEditorMode('yaml')
        break
      case 'more':
        // TODO: Show context menu
        break
    }
  }

  const handleGroupAction = (action: string, groupId: string) => {
    switch (action) {
      case 'add':
        createNewSchemaInGroup(groupId)
        break
      case 'settings':
        // TODO: Show group settings
        break
    }
  }

  const createNewSchemaInGroup = (groupId: string) => {
    const name = prompt('New schema name')?.trim()
    if (!name) return
    
    const newSchemaId = `${groupId}_${Date.now()}`
    const newSchema = {
      id: newSchemaId,
      name,
      kind: 'structured' as const,
      groupId,
      content: `__table_name__: ${name}\n__description__: New schema\n\nid:\n  type: integer\n  constraints:\n    primary_key: true\nname:\n  type: string\n`,
      metadata: {
        fieldCount: 2,
        lastModified: new Date(),
        status: 'draft' as const
      }
    }
    
    setSchemaGroups(prev => prev.map(group => 
      group.id === groupId 
        ? { ...group, schemas: [...group.schemas, newSchema] }
        : group
    ))
    
    setSelectedSchema(newSchemaId)
  }

  const createNewGroup = () => {
    const name = prompt('New group name')?.trim()
    if (!name) return
    
    const newGroup = {
      id: `group_${Date.now()}`,
      name,
      icon: '📁',
      description: `${name} schemas`,
      expanded: true,
      schemas: []
    }
    
    setSchemaGroups(prev => [...prev, newGroup])
  }

  const handleDatabaseImport = (tables: any[]) => {
    // Create new schemas in the first group or create a new "Imported" group
    const importGroupId = 'imported'
    const newSchemas = tables.map(table => ({
      id: `${importGroupId}_${table.name}`,
      name: table.name,
      kind: 'structured' as const,
      groupId: importGroupId,
      content: databaseTableToYaml(table),
      metadata: {
        fieldCount: table.columns.length,
        lastModified: new Date(),
        status: 'ready' as const
      }
    }))
    
    // Check if imported group exists, if not create it
    const hasImportedGroup = schemaGroups.some(g => g.id === importGroupId)
    
    if (!hasImportedGroup) {
      const importedGroup = {
        id: importGroupId,
        name: 'Imported',
        icon: '🔌',
        description: 'Schemas imported from database',
        expanded: true,
        schemas: newSchemas
      }
      setSchemaGroups(prev => [...prev, importedGroup])
    } else {
      setSchemaGroups(prev => prev.map(group => 
        group.id === importGroupId 
          ? { ...group, schemas: [...group.schemas, ...newSchemas] }
          : group
      ))
    }
    
    setShowImportModal(false)
    
    // Select the first imported schema
    if (newSchemas.length > 0) {
      setSelectedSchema(newSchemas[0].id)
    }
  }

  const deleteCurrentSchema = () => {
    if (!currentSchema) return
    if (confirm(`Delete schema "${currentSchema.name}"?`)) {
      setSchemaGroups(prev => prev.map(group => ({
        ...group,
        schemas: group.schemas.filter(s => s.id !== currentSchema.id)
      })))
      setSelectedSchema(null)
    }
  }

  return (
    <div style={{ display: 'grid', gridTemplateRows: 'auto 1fr', height: '100vh' }}>
      {/* Page Header */}
      <div style={{ 
        padding: '16px 20px', 
        borderBottom: '1px solid var(--border)',
        background: 'var(--panel)',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between'
      }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 16 }}>
          <h2 style={{ margin: 0, display: 'flex', alignItems: 'center', gap: 8 }}>
            📋 Schemas
          </h2>
          {currentSchema && (
            <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
               <span style={{ color: 'var(--muted)' }}>&gt;</span>
               <span style={{ fontWeight: 600 }}>{currentGroup?.name}</span>
               <span style={{ color: 'var(--muted)' }}>&gt;</span>
              <span style={{ fontWeight: 600 }}>{currentSchema.name}</span>
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
          )}
        </div>
        
        <div style={{ display: 'flex', gap: 8 }}>
          <button 
            className="btn secondary" 
            onClick={() => setShowImportModal(true)}
            style={{ padding: '8px 12px', fontSize: '0.9rem' }}
          >
            🔌 Import Database
          </button>
          <button 
            className="btn secondary" 
            onClick={createNewGroup}
            style={{ padding: '8px 12px', fontSize: '0.9rem' }}
          >
            📁 New Group
          </button>
          {currentSchema && (
            <>
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
              <button 
                className="btn secondary"
                onClick={deleteCurrentSchema}
                style={{ padding: '8px 12px', fontSize: '0.9rem' }}
              >
                🗑️ Delete
              </button>
            </>
          )}
        </div>
      </div>

      {/* Main Content */}
      <div style={{ display: 'grid', gridTemplateColumns: '300px 1fr', height: '100%', overflow: 'hidden' }}>
        {/* Tree View Sidebar */}
        <div style={{ 
          borderRight: '1px solid var(--border)', 
          background: 'var(--panel-2)',
          overflow: 'hidden',
          display: 'flex',
          flexDirection: 'column'
        }}>
          <div style={{ 
            padding: '12px 16px', 
            borderBottom: '1px solid var(--border)',
            fontWeight: 600,
            background: 'var(--panel)'
          }}>
            Schema Groups
          </div>
          <TreeView
            groups={schemaGroups}
            selectedSchema={selectedSchema}
            onSelectSchema={setSelectedSchema}
            onToggleGroup={handleToggleGroup}
            onSchemaAction={handleSchemaAction}
            onGroupAction={handleGroupAction}
          />
        </div>

        {/* Editor Area */}
        <div style={{ display: 'flex', flexDirection: 'column', overflow: 'hidden' }}>
          {/* Validation Errors */}
          {validationErrors.length > 0 && (
            <div className="panel" style={{ 
              margin: '16px 16px 0 16px',
              padding: 12, 
              background: 'rgba(239, 68, 68, 0.1)', 
              border: '1px solid rgba(239, 68, 68, 0.2)' 
            }}>
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
          <div style={{ flex: 1, overflow: 'hidden', padding: 16 }}>
            {!currentSchema ? (
              <div className="panel" style={{ 
                height: '100%',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                flexDirection: 'column',
                textAlign: 'center'
              }}>
                <div style={{ fontSize: '3rem', marginBottom: 16 }}>📋</div>
                <h3 style={{ margin: '0 0 8px 0' }}>No Schema Selected</h3>
                <p className="muted" style={{ marginBottom: 16 }}>
                  Select a schema from the tree view or create a new one to get started.
                </p>
                <div style={{ display: 'flex', gap: 8 }}>
                  <button className="btn" onClick={createNewGroup}>
                    📁 Create Group
                  </button>
                  <button className="btn secondary" onClick={() => setShowImportModal(true)}>
                    🔌 Import Database
                  </button>
                </div>
              </div>
            ) : currentSchema.kind === 'template' ? (
              // Template schemas always use YAML editor
              <div style={{ height: '100%', display: 'flex', flexDirection: 'column' }}>
                <div className="panel" style={{ 
                  padding: 12, 
                  marginBottom: 16, 
                  background: 'rgba(139, 92, 246, 0.1)', 
                  border: '1px solid rgba(139, 92, 246, 0.2)' 
                }}>
                  <div style={{ display: 'flex', alignItems: 'center', gap: 8, fontSize: '0.9rem' }}>
                    <span>📄</span>
                    <strong>Template Schema</strong> - Templates use YAML configuration for document generation
                  </div>
                </div>
                <textarea
                  value={currentSchema.content}
                  onChange={(e) => updateCurrentSchema(e.target.value)}
                  className="textarea"
                  style={{ 
                    flex: 1,
                    fontFamily: 'ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace', 
                    fontSize: 13,
                    resize: 'none'
                  }}
                  placeholder="Enter template configuration..."
                />
              </div>
            ) : editorMode === 'spreadsheet' ? (
              // Spreadsheet editor for structured schemas
              <SpreadsheetSchemaEditor
                schemaName={currentSchema.name}
                onFieldsChange={handleSpreadsheetChange}
              />
            ) : (
              // YAML editor for structured schemas
              <div style={{ height: '100%', display: 'flex', flexDirection: 'column' }}>
                <div className="panel" style={{ 
                  padding: 12, 
                  marginBottom: 16, 
                  background: 'rgba(59, 130, 246, 0.1)', 
                  border: '1px solid rgba(59, 130, 246, 0.2)' 
                }}>
                  <div style={{ display: 'flex', alignItems: 'center', gap: 8, fontSize: '0.9rem' }}>
                    <span>📝</span>
                    <strong>YAML Editor</strong> - Advanced configuration with full control
                  </div>
                </div>
                <textarea
                  value={currentSchema.content}
                  onChange={(e) => updateCurrentSchema(e.target.value)}
                  className="textarea"
                  style={{ 
                    flex: 1,
                    fontFamily: 'ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace', 
                    fontSize: 13,
                    resize: 'none'
                  }}
                  placeholder="Enter YAML schema definition..."
                />
              </div>
            )}
          </div>
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
