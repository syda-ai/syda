import { useMemo, useState } from 'react'
import { useAppState } from '../../store/AppState'
import TreeView from '../schemas/TreeView'

function downloadCsv(name: string, rows: any[]) {
  if (!rows?.length) return
  const headers = Object.keys(rows[0])
  const csv = [headers.join(','), ...rows.map(r => headers.map(h => JSON.stringify(r[h] ?? '')).join(','))].join('\n')
  const blob = new Blob([csv], { type: 'text/csv;charset=utf-8;' })
  const url = URL.createObjectURL(blob)
  const a = document.createElement('a')
  a.href = url
  a.download = `${name}.csv`
  a.click()
  URL.revokeObjectURL(url)
}

export default function ResultsPage() {
  const { results, schemaGroups, setSchemaGroups, selectedSchema, setSelectedSchema } = useAppState()
  const names = Object.keys(results)
  const [active, setActive] = useState<string | null>(names[0] || null)
  const [useDummy, setUseDummy] = useState<boolean>(false)

  const rows = results[active || ''] || []
  const headers = useMemo(() => (rows[0] ? Object.keys(rows[0]) : []), [rows])

  // Dummy table from provided YAML (Category)
  const dummyHeaders = ['id', 'name', 'parent_id']
  const dummyRows = [
    { id: 1, name: 'Electronics', parent_id: 0 },
    { id: 2, name: 'Phones', parent_id: 1 },
    { id: 3, name: 'Accessories', parent_id: 1 }
  ]

  const findSchemaById = (schemaId: string | null) => {
    if (!schemaId) return null
    for (const group of schemaGroups) {
      const s = group.schemas.find(sc => sc.id === schemaId)
      if (s) return s
    }
    return null
  }

  const handleToggleGroup = (groupId: string) => {
    setSchemaGroups(prev => prev.map(group =>
      group.id === groupId ? { ...group, expanded: !group.expanded } : group
    ))
  }

  const handleSchemaAction = (_action: string, schemaId: string) => {
    setSelectedSchema(schemaId)
    const schema = findSchemaById(schemaId)
    const schemaName = schema?.name || ''
    const isCategoryOrProduct = /category|product/i.test(schemaName)
    setUseDummy(isCategoryOrProduct)
    if (schema) {
      if (!isCategoryOrProduct && results[schema.name]) {
        setActive(schema.name)
      } else {
        // Keep active as-is; dummy rendering will take over
      }
    }
  }

  const handleGroupAction = (action: string, groupId: string) => {
    if (action !== 'add') return
    const name = prompt('New schema name')?.trim()
    if (!name) return
    const newSchemaId = `${groupId}_${Date.now()}`
    const newSchema = {
      id: newSchemaId,
      name,
      kind: 'structured' as const,
      groupId,
      content: `__table_name__: ${name}\n__description__: New schema\n`,
      metadata: {
        fieldCount: 0,
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

  // Sync selection to results names on initial mount or when selection changes
  const currentSelected = findSchemaById(selectedSchema)
  if (currentSelected && active !== currentSelected.name && results[currentSelected.name]) {
    // Prefer schema selection if it matches an existing results set
    // Note: This is a soft sync; user can still pick any results via dropdown
  }

  return (
    <div className="page-container" style={{ display: 'grid', gridTemplateColumns: '300px 1fr', height: '100%', overflow: 'hidden' }}>
      {/* Left Sidebar - Schema Groups */}
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
          onSelectSchema={(id) => handleSchemaAction('select', id)}
          onToggleGroup={handleToggleGroup}
          onSchemaAction={handleSchemaAction}
          onGroupAction={handleGroupAction}
        />
      </div>

      {/* Right Content - Results */}
      <div style={{ display: 'flex', flexDirection: 'column', minWidth: 0 }}>
        <div className="row gap-12 items-center" style={{ padding: '16px 20px', borderBottom: '1px solid var(--border)', background: 'var(--panel)' }}>
          <h2 style={{ margin: 0 }} className="page-title-gradient">📊 Results</h2>
          <select className="select" value={active || ''} onChange={(e) => { setActive(e.target.value || null); setUseDummy(false) }}>
            {names.map(n => <option key={n} value={n}>{n}</option>)}
          </select>
          <button 
            className="btn" 
            disabled={!(useDummy ? dummyRows.length : rows.length)} 
            onClick={() => {
              const name = useDummy ? (findSchemaById(selectedSchema)?.name || 'Category') : (active || 'results')
              downloadCsv(name, useDummy ? dummyRows : rows)
            }}
          >
            Download CSV
          </button>
        </div>
        <div className="panel overflow-auto" style={{ margin: 16, flex: 1 }}>
          <table className="table table-full">
            <thead>
              <tr>
                {(useDummy ? dummyHeaders : headers).map(h => (
                  <th key={h} style={{ textAlign: 'left' }}>{h}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {(useDummy ? dummyRows : rows).map((r, i) => (
                <tr key={i}>
                  {(useDummy ? dummyHeaders : headers).map(h => (
                    <td key={h}>{String(r[h] ?? '')}</td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  )
}



