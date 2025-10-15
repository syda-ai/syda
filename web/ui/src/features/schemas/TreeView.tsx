import { useState } from 'react'
import type { SchemaGroup, SchemaEntry } from '../../store/AppState'

type TreeViewProps = {
  groups: SchemaGroup[]
  selectedSchema: string | null
  onSelectSchema: (schemaId: string) => void
  onToggleGroup: (groupId: string) => void
  onSchemaAction: (action: string, schemaId: string) => void
  onGroupAction: (action: string, groupId: string) => void
}

type TreeGroupProps = {
  group: SchemaGroup
  selectedSchema: string | null
  onSelectSchema: (schemaId: string) => void
  onToggleGroup: (groupId: string) => void
  onSchemaAction: (action: string, schemaId: string) => void
  onGroupAction: (action: string, groupId: string) => void
}

type TreeSchemaProps = {
  schema: SchemaEntry
  isSelected: boolean
  onSelect: (schemaId: string) => void
  onAction: (action: string, schemaId: string) => void
}

function TreeSchema({ schema, isSelected, onSelect, onAction }: TreeSchemaProps) {
  const [showActions, setShowActions] = useState(false)

  const getStatusIcon = (status?: string) => {
    switch (status) {
      case 'ready': return '✅'
      case 'error': return '❌'
      case 'draft': return '📝'
      default: return ''
    }
  }

  const getSchemaIcon = (kind: string) => {
    return kind === 'template' ? '📄' : '🗂️'
  }

  return (
    <div 
      className={`tree-schema ${isSelected ? 'selected' : ''}`}
      onClick={() => onSelect(schema.id)}
      onMouseEnter={() => setShowActions(true)}
      onMouseLeave={() => setShowActions(false)}
      style={{
        display: 'flex',
        alignItems: 'center',
        padding: '6px 12px 6px 32px',
        cursor: 'pointer',
        background: isSelected ? 'rgba(59, 130, 246, 0.15)' : 'transparent',
        borderRadius: '4px',
        margin: '1px 8px',
        transition: 'all 0.2s ease',
        fontSize: '0.9rem'
      }}
    >
      <span style={{ marginRight: 8, fontSize: '0.9rem' }}>
        {getSchemaIcon(schema.kind)}
      </span>
      <span style={{ flex: 1, fontWeight: isSelected ? 600 : 400 }}>
        {schema.name}
      </span>
      <span style={{ fontSize: '0.8rem', color: 'var(--muted)', marginRight: 8 }}>
        {schema.metadata?.fieldCount || 0} fields
      </span>
      <span style={{ marginRight: 8 }}>
        {getStatusIcon(schema.metadata?.status)}
      </span>
      
      {showActions && (
        <div style={{ display: 'flex', gap: 4 }}>
          {schema.kind === 'structured' && (
            <button
              className="btn secondary"
              onClick={(e) => {
                e.stopPropagation()
                onAction('spreadsheet', schema.id)
              }}
              style={{ padding: '2px 6px', fontSize: '0.7rem' }}
              title="Open in Spreadsheet"
            >
              📊
            </button>
          )}
          <button
            className="btn secondary"
            onClick={(e) => {
              e.stopPropagation()
              onAction('yaml', schema.id)
            }}
            style={{ padding: '2px 6px', fontSize: '0.7rem' }}
            title="Open in YAML"
          >
            📝
          </button>
          <button
            className="btn secondary"
            onClick={(e) => {
              e.stopPropagation()
              onAction('description', schema.id)
            }}
            style={{ padding: '2px 6px', fontSize: '0.7rem' }}
            title="Edit Description"
          >
            ℹ️
          </button>
          <button
            className="btn secondary"
            onClick={(e) => {
              e.stopPropagation()
              onAction('more', schema.id)
            }}
            style={{ padding: '2px 6px', fontSize: '0.7rem' }}
            title="More actions"
          >
            ⚙️
          </button>
        </div>
      )}
    </div>
  )
}

function TreeGroup({ group, selectedSchema, onSelectSchema, onToggleGroup, onSchemaAction, onGroupAction }: TreeGroupProps) {
  return (
    <div className="tree-group" style={{ marginBottom: 4 }}>
      <div 
        className="tree-group-header"
        onClick={() => onToggleGroup(group.id)}
        style={{
          display: 'flex',
          alignItems: 'center',
          padding: '8px 12px',
          cursor: 'pointer',
          background: 'var(--panel-2)',
          borderRadius: '6px',
          margin: '2px 8px',
          transition: 'all 0.2s ease',
          fontWeight: 600
        }}
      >
        <span style={{ 
          marginRight: 8, 
          fontSize: '0.8rem', 
          color: 'var(--muted)',
          transition: 'transform 0.2s ease',
          transform: group.expanded ? 'rotate(90deg)' : 'rotate(0deg)'
        }}>
          ▶
        </span>
        <span style={{ marginRight: 8, fontSize: '1rem' }}>
          {group.icon}
        </span>
        <span style={{ flex: 1 }}>
          {group.name}
        </span>
        <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginRight: 8 }}>
          <button
            onClick={(e) => {
              e.stopPropagation()
              onGroupAction('model', group.id)
            }}
            style={{ 
              fontSize: '0.7rem', 
              color: group.modelConfig ? 'var(--primary)' : 'var(--muted)', 
              background: group.modelConfig ? 'rgba(59, 130, 246, 0.1)' : 'rgba(107, 114, 128, 0.1)',
              padding: '2px 6px',
              borderRadius: '4px',
              fontWeight: 600,
              border: 'none',
              cursor: 'pointer',
              transition: 'all 0.2s ease'
            }}
            onMouseEnter={(e) => {
              e.currentTarget.style.background = group.modelConfig ? 'rgba(59, 130, 246, 0.2)' : 'rgba(107, 114, 128, 0.2)'
              e.currentTarget.style.transform = 'scale(1.05)'
            }}
            onMouseLeave={(e) => {
              e.currentTarget.style.background = group.modelConfig ? 'rgba(59, 130, 246, 0.1)' : 'rgba(107, 114, 128, 0.1)'
              e.currentTarget.style.transform = 'scale(1)'
            }}
            title="Configure AI model for this group"
          >
            {group.modelConfig ? (
              group.modelConfig.model_name.includes('sonnet') ? 'Sonnet' :
              group.modelConfig.model_name.includes('haiku') ? 'Haiku' :
              group.modelConfig.model_name.includes('opus') ? 'Opus' :
              group.modelConfig.model_name.includes('gpt-4') ? 'GPT-4' :
              group.modelConfig.model_name.includes('gemini') ? 'Gemini' :
              'Custom'
            ) : (
              'No Model'
            )}
          </button>
          <span style={{ fontSize: '0.8rem', color: 'var(--muted)' }}>
            ({group.schemas.length})
          </span>
        </div>
        
      </div>
      
      {group.expanded && (
        <div className="tree-group-children" style={{ marginTop: 4 }}>
          {group.schemas.map(schema => (
            <TreeSchema
              key={schema.id}
              schema={schema}
              isSelected={selectedSchema === schema.id}
              onSelect={onSelectSchema}
              onAction={onSchemaAction}
            />
          ))}
          
          {group.schemas.length === 0 && (
            <div style={{ 
              padding: '12px 32px', 
              color: 'var(--muted)', 
              fontSize: '0.85rem',
              fontStyle: 'italic'
            }}>
              No schemas in this group
            </div>
          )}
        </div>
      )}
    </div>
  )
}

export default function TreeView({ 
  groups, 
  selectedSchema, 
  onSelectSchema, 
  onToggleGroup, 
  onSchemaAction, 
  onGroupAction 
}: TreeViewProps) {
  return (
    <div className="tree-view" style={{ 
      padding: '8px 0',
      height: '100%',
      overflowY: 'auto'
    }}>
      {groups.map(group => (
        <TreeGroup
          key={group.id}
          group={group}
          selectedSchema={selectedSchema}
          onSelectSchema={onSelectSchema}
          onToggleGroup={onToggleGroup}
          onSchemaAction={onSchemaAction}
          onGroupAction={onGroupAction}
        />
      ))}
      
      {groups.length === 0 && (
        <div style={{ 
          padding: 40, 
          textAlign: 'center', 
          color: 'var(--muted)' 
        }}>
          <div style={{ fontSize: '2rem', marginBottom: 8 }}>📋</div>
          <div style={{ fontSize: '0.9rem' }}>No schema groups yet</div>
        </div>
      )}
    </div>
  )
}
