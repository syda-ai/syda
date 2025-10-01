import { useMemo, useState } from 'react'
import { useAppState } from '../../store/AppState'

type Row = Record<string, any>

function simulateRows(schemaName: string, count: number): Row[] {
  const rows: Row[] = []
  for (let i = 1; i <= count; i++) {
    rows.push({ id: i, name: `${schemaName} ${i}`, description: `Mock ${schemaName} row ${i}` })
  }
  return rows
}

export default function RunPage() {
  const { schemas, setResults } = useAppState()
  const [selected, setSelected] = useState<Record<string, boolean>>(() => Object.fromEntries(schemas.map(s => [s.name, s.kind !== 'template'])))
  const [sampleSizes, setSampleSizes] = useState<Record<string, number>>(() => Object.fromEntries(schemas.map(s => [s.name, s.kind === 'template' ? 5 : 10])))
  const [running, setRunning] = useState(false)
  const selectable = useMemo(() => schemas.map(s => ({ name: s.name, kind: s.kind })), [schemas])

  return (
    <div className="fade-in" style={{ display: 'grid', gap: 24, width: 800, maxWidth: 800 }}>
    <div className="fade-in" style={{ display: 'grid', gap: 16, width: '100%', maxWidth: '100%' }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
        <h2 style={{ margin: 0, background: 'linear-gradient(135deg, var(--primary-light), var(--accent-light))', backgroundClip: 'text', WebkitBackgroundClip: 'text', color: 'transparent' }}>⚙️ Run Generation</h2>
        <span className="muted">(Mock Mode)</span>
      </div>
      <div className="panel">
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', padding: 10, borderBottom: '1px solid #1f2937', fontWeight: 700 }}>
          <div>Schema</div>
          <div>Include</div>
          <div>Sample Size</div>
        </div>
        {selectable.map(row => (
          <div key={row.name} style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', padding: 10, borderTop: '1px solid #122033' }}>
            <div>{row.name} <span className="muted">({row.kind})</span></div>
            <div>
              <input type="checkbox" checked={!!selected[row.name]} onChange={(e) => setSelected(prev => ({ ...prev, [row.name]: e.target.checked }))} />
            </div>
            <div>
              <input className="input" type="number" min={1} style={{ width: 120 }} value={sampleSizes[row.name] || 0} onChange={(e) => setSampleSizes(prev => ({ ...prev, [row.name]: Number(e.target.value) }))} />
            </div>
          </div>
        ))}
      </div>
      <div style={{ display: 'flex', gap: 8 }}>
        <button className="btn" disabled={running} onClick={async () => {
          setRunning(true)
          await new Promise(r => setTimeout(r, 500))
          setResults(prev => {
            const next = { ...prev }
            Object.entries(selected).forEach(([name, inc]) => {
              if (inc) next[name] = simulateRows(name, sampleSizes[name] || 1)
            })
            return next
          })
          setRunning(false)
          alert('Mock generation complete')
        }}>Start Run</button>
        {running && <span>Running…</span>}
      </div>
    </div>
    </div>
  )
}



