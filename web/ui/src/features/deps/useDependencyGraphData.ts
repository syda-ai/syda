import { useQuery } from '@tanstack/react-query'
//import { apiClient } from '../../lib/apiClient'

export function useDependencyGraphData() {
  return useQuery({
    queryKey: ['dependency-graph'],
    queryFn: async () => {
      // Placeholder: call API when available
      try {
        const res = await apiClient('/api/dependencies/graph', { method: 'GET' })
        if (res?.nodes) return res
      } catch {}
      // Fallback demo data
      return {
        nodes: [
          { id: 'Category', label: 'Category', kind: 'structured', rank: 1 },
          { id: 'Product', label: 'Product', kind: 'structured', rank: 2 },
          { id: 'ProductCatalog', label: 'ProductCatalog', kind: 'template', rank: 3 },
        ],
        edges: [
          { id: 'product.category_id->Category.id', source: 'Product', target: 'Category', kind: 'foreign_key', fks: [{ from: 'category_id', to: 'id' }], status: 'ok' },
          { id: 'catalog.depends->Product', source: 'ProductCatalog', target: 'Product', kind: 'depends_on', status: 'ok' },
          { id: 'catalog.depends->Category', source: 'ProductCatalog', target: 'Category', kind: 'depends_on', status: 'ok' },
        ],
        order: ['Category', 'Product', 'ProductCatalog'],
        issues: { cycles: [], missing: [] },
      }
    },
  })
}



