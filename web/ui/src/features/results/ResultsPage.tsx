import { useMemo, useState } from 'react'
import { useAppState } from '../../store/AppState'

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
  const { results } = useAppState()
  const names = Object.keys(results)
  const [active, setActive] = useState<string | null>(names[0] || null)
  const rows = results[active || ''] || []
  const headers = useMemo(() => (rows[0] ? Object.keys(rows[0]) : []), [rows])

  return (
    <div className="fade-in page-container centered-800">
    <div className="fade-in page-container gap-16 w-full" style={{ maxWidth: '100%' }}>
      <div className="row gap-12 items-center">
        <h2 style={{ margin: 0 }} className="page-title-gradient">📊 Results</h2>
        <select className="select" value={active || ''} onChange={(e) => setActive(e.target.value || null)}>
          {names.map(n => <option key={n} value={n}>{n}</option>)}
        </select>
        <button className="btn" disabled={!rows.length} onClick={() => active && downloadCsv(active, rows)}>Download CSV</button>
      </div>
      <div className="panel overflow-auto">
        <table className="table table-full">
          <thead>
            <tr>
              {headers.map(h => (
                <th key={h} style={{ textAlign: 'left' }}>{h}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {rows.map((r, i) => (
              <tr key={i}>
                {headers.map(h => (
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



