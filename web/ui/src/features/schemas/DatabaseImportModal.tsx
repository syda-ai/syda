import { useState } from 'react'

type DatabaseConnection = {
  type: 'postgresql' | 'mysql' | 'sqlite' | 'sqlserver'
  host?: string
  port?: number
  database: string
  username?: string
  password?: string
  file?: string // for SQLite
}

type DatabaseTable = {
  name: string
  schema?: string
  columns: Array<{
    name: string
    type: string
    nullable: boolean
    primaryKey: boolean
    foreignKey?: {
      table: string
      column: string
    }
  }>
  selected: boolean
}

type DatabaseImportModalProps = {
  onImport: (tables: DatabaseTable[]) => void
  onClose: () => void
}

export default function DatabaseImportModal({ onImport, onClose }: DatabaseImportModalProps) {
  const [step, setStep] = useState<'connect' | 'select' | 'importing'>('connect')
  const [connection, setConnection] = useState<DatabaseConnection>({
    type: 'postgresql',
    host: 'localhost',
    port: 5432,
    database: '',
    username: '',
    password: ''
  })
  const [tables, setTables] = useState<DatabaseTable[]>([])
  const [connecting, setConnecting] = useState(false)
  const [error, setError] = useState<string | null>(null)

  // Mock database connection and table discovery
  const handleConnect = async () => {
    setConnecting(true)
    setError(null)
    
    try {
      // Simulate API call delay
      await new Promise(resolve => setTimeout(resolve, 1500))
      
      // Mock discovered tables
      const mockTables: DatabaseTable[] = [
        {
          name: 'users',
          columns: [
            { name: 'id', type: 'integer', nullable: false, primaryKey: true },
            { name: 'email', type: 'varchar(255)', nullable: false, primaryKey: false },
            { name: 'password_hash', type: 'varchar(255)', nullable: false, primaryKey: false },
            { name: 'first_name', type: 'varchar(100)', nullable: true, primaryKey: false },
            { name: 'last_name', type: 'varchar(100)', nullable: true, primaryKey: false },
            { name: 'created_at', type: 'timestamp', nullable: false, primaryKey: false },
            { name: 'updated_at', type: 'timestamp', nullable: false, primaryKey: false }
          ],
          selected: true
        },
        {
          name: 'products',
          columns: [
            { name: 'id', type: 'integer', nullable: false, primaryKey: true },
            { name: 'name', type: 'varchar(255)', nullable: false, primaryKey: false },
            { name: 'description', type: 'text', nullable: true, primaryKey: false },
            { name: 'price', type: 'decimal(10,2)', nullable: false, primaryKey: false },
            { name: 'category_id', type: 'integer', nullable: true, primaryKey: false, foreignKey: { table: 'categories', column: 'id' } },
            { name: 'created_at', type: 'timestamp', nullable: false, primaryKey: false }
          ],
          selected: true
        },
        {
          name: 'categories',
          columns: [
            { name: 'id', type: 'integer', nullable: false, primaryKey: true },
            { name: 'name', type: 'varchar(100)', nullable: false, primaryKey: false },
            { name: 'parent_id', type: 'integer', nullable: true, primaryKey: false, foreignKey: { table: 'categories', column: 'id' } }
          ],
          selected: true
        },
        {
          name: 'orders',
          columns: [
            { name: 'id', type: 'integer', nullable: false, primaryKey: true },
            { name: 'user_id', type: 'integer', nullable: false, primaryKey: false, foreignKey: { table: 'users', column: 'id' } },
            { name: 'total_amount', type: 'decimal(10,2)', nullable: false, primaryKey: false },
            { name: 'status', type: 'varchar(50)', nullable: false, primaryKey: false },
            { name: 'created_at', type: 'timestamp', nullable: false, primaryKey: false }
          ],
          selected: false
        },
        {
          name: 'order_items',
          columns: [
            { name: 'id', type: 'integer', nullable: false, primaryKey: true },
            { name: 'order_id', type: 'integer', nullable: false, primaryKey: false, foreignKey: { table: 'orders', column: 'id' } },
            { name: 'product_id', type: 'integer', nullable: false, primaryKey: false, foreignKey: { table: 'products', column: 'id' } },
            { name: 'quantity', type: 'integer', nullable: false, primaryKey: false },
            { name: 'price', type: 'decimal(10,2)', nullable: false, primaryKey: false }
          ],
          selected: false
        }
      ]
      
      setTables(mockTables)
      setStep('select')
    } catch (err) {
      setError('Failed to connect to database. Please check your connection details.')
    } finally {
      setConnecting(false)
    }
  }

  const handleImport = async () => {
    setStep('importing')
    
    // Simulate import process
    await new Promise(resolve => setTimeout(resolve, 2000))
    
    const selectedTables = tables.filter(t => t.selected)
    onImport(selectedTables)
  }

  const toggleTable = (tableName: string) => {
    setTables(tables.map(t => 
      t.name === tableName ? { ...t, selected: !t.selected } : t
    ))
  }

  const toggleAllTables = () => {
    const allSelected = tables.every(t => t.selected)
    setTables(tables.map(t => ({ ...t, selected: !allSelected })))
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
      <div className="panel" style={{ width: 700, maxHeight: '80vh', overflow: 'hidden', display: 'flex', flexDirection: 'column' }}>
        {/* Header */}
        <div style={{ padding: 24, borderBottom: '1px solid var(--border)' }}>
          <h3 style={{ margin: '0 0 8px 0', display: 'flex', alignItems: 'center', gap: 8 }}>
            🔌 Import Database Schema
          </h3>
          <div style={{ display: 'flex', alignItems: 'center', gap: 16, fontSize: '0.9rem', color: 'var(--muted)' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: 4 }}>
              <div style={{ 
                width: 8, 
                height: 8, 
                borderRadius: '50%', 
                background: step === 'connect' ? 'var(--primary)' : step === 'select' || step === 'importing' ? 'var(--success)' : 'var(--border)' 
              }} />
              Connect
            </div>
            <div style={{ width: 20, height: 1, background: 'var(--border)' }} />
            <div style={{ display: 'flex', alignItems: 'center', gap: 4 }}>
              <div style={{ 
                width: 8, 
                height: 8, 
                borderRadius: '50%', 
                background: step === 'select' ? 'var(--primary)' : step === 'importing' ? 'var(--success)' : 'var(--border)' 
              }} />
              Select Tables
            </div>
            <div style={{ width: 20, height: 1, background: 'var(--border)' }} />
            <div style={{ display: 'flex', alignItems: 'center', gap: 4 }}>
              <div style={{ 
                width: 8, 
                height: 8, 
                borderRadius: '50%', 
                background: step === 'importing' ? 'var(--primary)' : 'var(--border)' 
              }} />
              Import
            </div>
          </div>
        </div>

        {/* Content */}
        <div style={{ flex: 1, overflow: 'auto', padding: 24 }}>
          {step === 'connect' && (
            <div style={{ display: 'grid', gap: 16 }}>
              <div>
                <label style={{ display: 'block', marginBottom: 4, fontWeight: 600, fontSize: '0.9rem' }}>
                  Database Type
                </label>
                <select
                  className="select"
                  value={connection.type}
                  onChange={(e) => setConnection({ ...connection, type: e.target.value as any })}
                >
                  <option value="postgresql">PostgreSQL</option>
                  <option value="mysql">MySQL</option>
                  <option value="sqlite">SQLite</option>
                  <option value="sqlserver">SQL Server</option>
                </select>
              </div>

              {connection.type !== 'sqlite' ? (
                <div style={{ display: 'grid', gridTemplateColumns: '1fr 120px', gap: 12 }}>
                  <div>
                    <label style={{ display: 'block', marginBottom: 4, fontWeight: 600, fontSize: '0.9rem' }}>
                      Host
                    </label>
                    <input
                      className="input"
                      value={connection.host}
                      onChange={(e) => setConnection({ ...connection, host: e.target.value })}
                      placeholder="localhost"
                    />
                  </div>
                  <div>
                    <label style={{ display: 'block', marginBottom: 4, fontWeight: 600, fontSize: '0.9rem' }}>
                      Port
                    </label>
                    <input
                      className="input"
                      type="number"
                      value={connection.port}
                      onChange={(e) => setConnection({ ...connection, port: Number(e.target.value) })}
                      placeholder="5432"
                    />
                  </div>
                </div>
              ) : (
                <div>
                  <label style={{ display: 'block', marginBottom: 4, fontWeight: 600, fontSize: '0.9rem' }}>
                    Database File
                  </label>
                  <input
                    className="input"
                    value={connection.file || ''}
                    onChange={(e) => setConnection({ ...connection, file: e.target.value })}
                    placeholder="/path/to/database.db"
                  />
                </div>
              )}

              <div>
                <label style={{ display: 'block', marginBottom: 4, fontWeight: 600, fontSize: '0.9rem' }}>
                  Database Name
                </label>
                <input
                  className="input"
                  value={connection.database}
                  onChange={(e) => setConnection({ ...connection, database: e.target.value })}
                  placeholder="my_database"
                />
              </div>

              {connection.type !== 'sqlite' && (
                <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 12 }}>
                  <div>
                    <label style={{ display: 'block', marginBottom: 4, fontWeight: 600, fontSize: '0.9rem' }}>
                      Username
                    </label>
                    <input
                      className="input"
                      value={connection.username}
                      onChange={(e) => setConnection({ ...connection, username: e.target.value })}
                      placeholder="username"
                    />
                  </div>
                  <div>
                    <label style={{ display: 'block', marginBottom: 4, fontWeight: 600, fontSize: '0.9rem' }}>
                      Password
                    </label>
                    <input
                      className="input"
                      type="password"
                      value={connection.password}
                      onChange={(e) => setConnection({ ...connection, password: e.target.value })}
                      placeholder="password"
                    />
                  </div>
                </div>
              )}

              {error && (
                <div className="panel" style={{ padding: 12, background: 'rgba(239, 68, 68, 0.1)', border: '1px solid rgba(239, 68, 68, 0.2)' }}>
                  <div style={{ color: 'var(--danger)', fontSize: '0.9rem' }}>
                    {error}
                  </div>
                </div>
              )}
            </div>
          )}

          {step === 'select' && (
            <div style={{ display: 'grid', gap: 16 }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <div>
                  <h4 style={{ margin: 0 }}>Select Tables to Import</h4>
                  <p style={{ margin: '4px 0 0 0', fontSize: '0.9rem', color: 'var(--muted)' }}>
                    {tables.filter(t => t.selected).length} of {tables.length} tables selected
                  </p>
                </div>
                <button className="btn secondary" onClick={toggleAllTables}>
                  {tables.every(t => t.selected) ? 'Deselect All' : 'Select All'}
                </button>
              </div>

              <div style={{ maxHeight: '400px', overflow: 'auto' }}>
                {tables.map(table => (
                  <div key={table.name} className="panel" style={{ marginBottom: 12, padding: 16 }}>
                    <div style={{ display: 'flex', alignItems: 'center', gap: 12, marginBottom: 12 }}>
                      <input
                        type="checkbox"
                        checked={table.selected}
                        onChange={() => toggleTable(table.name)}
                      />
                      <div>
                        <div style={{ fontWeight: 600, display: 'flex', alignItems: 'center', gap: 8 }}>
                          🗂️ {table.name}
                        </div>
                        <div style={{ fontSize: '0.85rem', color: 'var(--muted)' }}>
                          {table.columns.length} columns
                        </div>
                      </div>
                    </div>
                    
                    {table.selected && (
                      <div style={{ fontSize: '0.85rem', marginLeft: 24 }}>
                        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: 8 }}>
                          {table.columns.slice(0, 6).map(col => (
                            <div key={col.name} style={{ display: 'flex', alignItems: 'center', gap: 4 }}>
                              {col.primaryKey && <span style={{ color: 'var(--primary)' }}>🔑</span>}
                              {col.foreignKey && <span style={{ color: 'var(--accent)' }}>🔗</span>}
                              <span>{col.name}</span>
                              <span style={{ color: 'var(--muted)' }}>({col.type})</span>
                            </div>
                          ))}
                          {table.columns.length > 6 && (
                            <div style={{ color: 'var(--muted)' }}>
                              +{table.columns.length - 6} more columns...
                            </div>
                          )}
                        </div>
                      </div>
                    )}
                  </div>
                ))}
              </div>
            </div>
          )}

          {step === 'importing' && (
            <div style={{ textAlign: 'center', padding: 40 }}>
              <div style={{ fontSize: '3rem', marginBottom: 16 }}>⚡</div>
              <h4 style={{ margin: '0 0 8px 0' }}>Importing Schemas...</h4>
              <p style={{ margin: 0, color: 'var(--muted)' }}>
                Converting database tables to Syda schemas
              </p>
              <div style={{ marginTop: 20 }}>
                <div style={{ 
                  width: '100%', 
                  height: 4, 
                  background: 'var(--border)', 
                  borderRadius: 2,
                  overflow: 'hidden'
                }}>
                  <div style={{ 
                    width: '60%', 
                    height: '100%', 
                    background: 'linear-gradient(90deg, var(--primary), var(--accent))',
                    animation: 'progress 2s ease-in-out infinite'
                  }} />
                </div>
              </div>
            </div>
          )}
        </div>

        {/* Footer */}
        <div style={{ padding: 24, borderTop: '1px solid var(--border)', display: 'flex', gap: 8, justifyContent: 'flex-end' }}>
          {step !== 'importing' && (
            <button className="btn secondary" onClick={onClose}>
              Cancel
            </button>
          )}
          
          {step === 'connect' && (
            <button 
              className="btn" 
              onClick={handleConnect}
              disabled={connecting || !connection.database}
            >
              {connecting ? 'Connecting...' : 'Connect & Discover Tables'}
            </button>
          )}
          
          {step === 'select' && (
            <button 
              className="btn" 
              onClick={handleImport}
              disabled={!tables.some(t => t.selected)}
            >
              Import {tables.filter(t => t.selected).length} Tables
            </button>
          )}
        </div>
      </div>

      <style>{`
        @keyframes progress {
          0% { transform: translateX(-100%); }
          50% { transform: translateX(0%); }
          100% { transform: translateX(100%); }
        }
      `}</style>
    </div>
  )
}
