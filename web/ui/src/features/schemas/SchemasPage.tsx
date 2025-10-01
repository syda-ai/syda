import { useMemo, useState } from 'react'
import { useAppState } from '../../store/AppState'

export default function SchemasPage() {
  const { schemas, setSchemas } = useAppState()
  const [selected, setSelected] = useState<string | null>(schemas[0]?.name || null)
  const current = useMemo(() => schemas.find(s => s.name === selected) || null, [schemas, selected])

  return (
    <div className="fade-in" style={{ display: 'grid', gap: 24, width: 800, maxWidth: 800 }}>
    <div className="fade-in" style={{ display: 'grid', gridTemplateColumns: '280px 1fr', gap: 20, width: '100%', maxWidth: '100%' }}>
      <div className="panel">
        <div style={{ padding: 16, fontWeight: 700, borderBottom: '1px solid #1f2937', position: 'relative', zIndex: 1 }}>📋 Schemas</div>
        <div>
          {schemas.map(s => (
            <div key={s.name} onClick={() => setSelected(s.name)} style={{ 
              padding: 12, 
              cursor: 'pointer', 
              background: selected === s.name ? 'linear-gradient(135deg, rgba(59, 130, 246, 0.15), rgba(139, 92, 246, 0.1))' : 'transparent', 
              borderTop: '1px solid #122033',
              transition: 'all 0.2s ease',
              position: 'relative',
              zIndex: 1
            }}>
              <div style={{ fontWeight: 600, display: 'flex', alignItems: 'center', gap: 8 }}>
                <span>{s.kind === 'template' ? '📄' : '🗂️'}</span>
                {s.name}
              </div>
              <div className="muted" style={{ fontSize: 12 }}>{s.kind}</div>
            </div>
          ))}
        </div>
      </div>
      <div style={{ display: 'grid', gap: 8 }}>
        <div style={{ display: 'flex', gap: 8, alignItems: 'center' }}>
          <h3 style={{ margin: 0 }}>{current?.name || 'No schema selected'}</h3>
          {current && <span className="muted">({current.kind})</span>}
        </div>
        <textarea
          value={current?.content || ''}
          onChange={(e) => {
            const v = e.target.value
            if (!current) return
            setSchemas(prev => prev.map(s => s.name === current.name ? { ...s, content: v } : s))
          }}
          className="textarea"
          style={{ width: '100%', height: 420, fontFamily: 'ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace', fontSize: 13 }}
        />
        <div style={{ display: 'flex', gap: 8 }}>
          <button className="btn"
            onClick={() => {
              const name = prompt('New schema name')?.trim()
              if (!name) return
              setSchemas(prev => ([...prev, { name, kind: 'structured', content: 'id:\n  type: integer\nname:\n  type: string\n' }]))
              setSelected(name)
            }}
          >Add Schema</button>
          {current && (
            <button className="btn secondary"
              onClick={() => {
                setSchemas(prev => prev.filter(s => s.name !== current.name))
                setSelected(null)
              }}
            >Delete</button>
          )}
        </div>
      </div>
    </div>
    </div>
  )
}



