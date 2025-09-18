import React, { createContext, useContext, useMemo, useState } from 'react'

export type SchemaEntry = {
  name: string
  kind: 'structured' | 'template'
  content: string
}

export type ResultsMap = Record<string, any[]>

type AppState = {
  schemas: SchemaEntry[]
  setSchemas: (updater: (prev: SchemaEntry[]) => SchemaEntry[]) => void
  results: ResultsMap
  setResults: React.Dispatch<React.SetStateAction<ResultsMap>>
}

const AppStateContext = createContext<AppState | null>(null)

function initialSchemas(): SchemaEntry[] {
  return [
    {
      name: 'Category',
      kind: 'structured',
      content: `__table_name__: Category
__description__: Retail product categories

id:
  type: integer
  constraints:
    primary_key: true
name:
  type: string
parent_id:
  type: integer
  description: Parent category id (0 for root)
`,
    },
    {
      name: 'Product',
      kind: 'structured',
      content: `__table_name__: Product
__description__: Retail products
__foreign_keys__:
  category_id: [Category, id]

id:
  type: integer
  constraints:
    primary_key: true
name:
  type: string
category_id:
  type: integer
price:
  type: float
`,
    },
    {
      name: 'ProductCatalog',
      kind: 'template',
      content: `__template__: true
__name__: ProductCatalog
__depends_on__: [Product, Category]
__template_source__: templates/product_catalog.html
__input_file_type__: html
__output_file_type__: pdf

product_name:
  type: string
category_name:
  type: string
product_price:
  type: float
`,
    },
  ]
}

export function AppStateProvider({ children }: { children: React.ReactNode }) {
  const [schemas, _setSchemas] = useState<SchemaEntry[]>(initialSchemas)
  const [results, setResults] = useState<ResultsMap>({})

  const setSchemas = (updater: (prev: SchemaEntry[]) => SchemaEntry[]) => {
    _setSchemas((prev) => updater(prev))
  }

  const value = useMemo<AppState>(() => ({ schemas, setSchemas, results, setResults }), [schemas, results])
  return <AppStateContext.Provider value={value}>{children}</AppStateContext.Provider>
}

export function useAppState() {
  const ctx = useContext(AppStateContext)
  if (!ctx) throw new Error('useAppState must be used within AppStateProvider')
  return ctx
}



