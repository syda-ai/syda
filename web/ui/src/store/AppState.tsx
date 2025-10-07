import React, { createContext, useContext, useMemo, useState } from 'react'

export type SchemaEntry = {
  id: string
  name: string
  kind: 'structured' | 'template'
  content: string
  groupId: string
  metadata?: {
    fieldCount?: number
    lastModified?: Date
    status?: 'draft' | 'ready' | 'error'
  }
}

export type SchemaGroup = {
  id: string
  name: string
  icon: string
  description?: string
  expanded: boolean
  schemas: SchemaEntry[]
}

export type ResultsMap = Record<string, any[]>

type AppState = {
  schemaGroups: SchemaGroup[]
  setSchemaGroups: (updater: (prev: SchemaGroup[]) => SchemaGroup[]) => void
  selectedSchema: string | null
  setSelectedSchema: (schemaId: string | null) => void
  results: ResultsMap
  setResults: React.Dispatch<React.SetStateAction<ResultsMap>>
  // Legacy support - will be removed
  schemas: SchemaEntry[]
  setSchemas: (updater: (prev: SchemaEntry[]) => SchemaEntry[]) => void
}

const AppStateContext = createContext<AppState | null>(null)

function initialSchemaGroups(): SchemaGroup[] {
  return [
    {
      id: 'ecommerce',
      name: 'E-commerce',
      icon: '🛍️',
      description: 'Online store schemas',
      expanded: true,
      schemas: [
        {
          id: 'category',
          name: 'Category',
          kind: 'structured',
          groupId: 'ecommerce',
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
          metadata: {
            fieldCount: 3,
            lastModified: new Date(),
            status: 'ready'
          }
        },
        {
          id: 'product',
          name: 'Product',
          kind: 'structured',
          groupId: 'ecommerce',
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
          metadata: {
            fieldCount: 4,
            lastModified: new Date(),
            status: 'ready'
          }
        },
        {
          id: 'product_catalog',
          name: 'ProductCatalog',
          kind: 'template',
          groupId: 'ecommerce',
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
          metadata: {
            fieldCount: 3,
            lastModified: new Date(),
            status: 'ready'
          }
        }
      ]
    },
    {
      id: 'analytics',
      name: 'Analytics',
      icon: '📊',
      description: 'Data tracking schemas',
      expanded: false,
      schemas: []
    }
  ]
}


export function AppStateProvider({ children }: { children: React.ReactNode }) {
  const [schemaGroups, _setSchemaGroups] = useState<SchemaGroup[]>(initialSchemaGroups)
  const [selectedSchema, setSelectedSchema] = useState<string | null>('category')
  const [results, setResults] = useState<ResultsMap>({})

  const setSchemaGroups = (updater: (prev: SchemaGroup[]) => SchemaGroup[]) => {
    _setSchemaGroups((prev) => updater(prev))
  }

  // Legacy support - flatten groups to schemas
  const schemas = useMemo(() => schemaGroups.flatMap(group => group.schemas), [schemaGroups])
  
  const setSchemas = (updater: (prev: SchemaEntry[]) => SchemaEntry[]) => {
    const newSchemas = updater(schemas)
    // Update groups with new schemas
    _setSchemaGroups(prev => prev.map(group => ({
      ...group,
      schemas: newSchemas.filter(schema => schema.groupId === group.id)
    })))
  }

  const value = useMemo<AppState>(() => ({ 
    schemaGroups, 
    setSchemaGroups, 
    selectedSchema, 
    setSelectedSchema,
    results, 
    setResults,
    schemas, 
    setSchemas 
  }), [schemaGroups, selectedSchema, results, schemas])
  
  return <AppStateContext.Provider value={value}>{children}</AppStateContext.Provider>
}

export function useAppState() {
  const ctx = useContext(AppStateContext)
  if (!ctx) throw new Error('useAppState must be used within AppStateProvider')
  return ctx
}



