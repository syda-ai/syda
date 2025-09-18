import React, { useCallback, useMemo } from 'react'
import ReactFlow, { Background, Controls, MiniMap, Position } from 'reactflow'
import 'reactflow/dist/style.css'
import dagre from 'dagre'
import { useDependencyGraphData } from './useDependencyGraphData'

const nodeWidth = 180
const nodeHeight = 46

function layoutGraph(nodes: any[], edges: any[]) {
  const g = new dagre.graphlib.Graph()
  g.setDefaultEdgeLabel(() => ({}))
  g.setGraph({ rankdir: 'LR', nodesep: 40, ranksep: 80 })

  nodes.forEach((n) => g.setNode(n.id, { width: nodeWidth, height: nodeHeight }))
  edges.forEach((e) => g.setEdge(e.source, e.target))
  dagre.layout(g)

  const laidOutNodes = nodes.map((n) => {
    const pos = g.node(n.id)
    return {
      ...n,
      position: { x: pos.x - nodeWidth / 2, y: pos.y - nodeHeight / 2 },
      targetPosition: Position.Left,
      sourcePosition: Position.Right,
    }
  })
  return { nodes: laidOutNodes, edges }
}

export default function DependencyGraph() {
  const { data, isLoading, error, refetch } = useDependencyGraphData()

  const { nodes, edges } = useMemo(() => {
    if (!data) return { nodes: [], edges: [] }

    const nodes = data.nodes.map((n: any) => ({
      id: n.id,
      data: { label: `${n.label}${Number.isFinite(n.rank) ? `  [${n.rank}]` : ''}` },
      style: {
        border: '1px solid #1f2937',
        borderRadius: 12,
        padding: '8px 12px',
        color: '#e5e7eb',
        background: n.kind === 'template' ? 'linear-gradient(180deg, rgba(167,139,250,0.2), rgba(59,130,246,0.12))' : 'linear-gradient(180deg, rgba(59,130,246,0.2), rgba(59,130,246,0.08))',
      },
      position: { x: 0, y: 0 },
      type: 'default',
    }))

    const edges = data.edges.map((e: any) => ({
      id: e.id,
      source: e.source,
      target: e.target,
      label: e.kind === 'foreign_key' ? (e.fks?.map((f: any) => `${f.from}→${f.to}`).join(', ') || 'FK') : 'depends',
      style: { stroke: e.status === 'missing' ? '#ef4444' : e.status === 'cycle' ? '#f59e0b' : '#3b82f6' },
      animated: e.status === 'cycle',
      markerEnd: { type: 'arrowclosed' as const },
      type: e.kind === 'depends_on' ? 'simplebezier' : 'default',
    }))

    return layoutGraph(nodes, edges)
  }, [data])

  const onRefresh = useCallback(() => { refetch() }, [refetch])

  if (isLoading) return <div className="muted">Loading dependency graph…</div>
  if (error) return <div className="danger">Failed to load graph</div>

  return (
    <div className="panel" style={{ height: 'calc(100vh - 140px)' }}>
      <div style={{ padding: '10px', borderBottom: '1px solid #1f2937', display: 'flex', gap: 8 }}>
        <button className="btn" onClick={onRefresh}>Refresh</button>
      </div>
      <ReactFlow nodes={nodes} edges={edges} fitView style={{ background: 'transparent' }}>
        <MiniMap nodeStrokeColor="#374151" nodeColor="#0b162a" maskColor="rgba(0,0,0,0.25)" />
        <Controls />
        <Background variant="dots" gap={16} size={1} color="#122033" />
      </ReactFlow>
      {data?.issues && (
        <div style={{ padding: 10, borderTop: '1px solid #1f2937' }}>
          {data.issues.cycles?.length ? (
            <div className="warn">Cycles: {data.issues.cycles.map((c: any) => c.join(' → ')).join(' | ')}</div>
          ) : null}
          {data.issues.missing?.length ? (
            <div className="danger">Missing: {data.issues.missing.join(', ')}</div>
          ) : null}
        </div>
      )}
    </div>
  )
}



