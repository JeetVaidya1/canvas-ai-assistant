import { useMemo } from 'react'
import { motion } from 'motion/react'
import { Network, ArrowRight } from 'lucide-react'
import type { ConceptGraph } from '@/lib/api'
import { scoreTone } from '@/lib/score'
import { Card } from '@/components/ui/Card'
import { EmptyState, ErrorState } from '@/components/ui/States'
import { SectionHead } from './SectionHead'

// Paper & Ink graph palette: hairline edges, white node sheets, ink-soft labels.
const NO_DATA_COLOR = '#d8d3c5'
const EDGE_COLOR = '#d8d3c5'
const NODE_FILL = '#ffffff'
const LABEL_COLOR = '#5d5850'
const MAX_BLOCKERS = 5

interface GraphNode {
  concept: string
  mastery_pct: number
  has_data: boolean
}

interface PositionedNode {
  node: GraphNode
  x: number
  y: number
}

interface PositionedEdge {
  from: { x: number; y: number }
  to: { x: number; y: number }
  prerequisite: string
  concept: string
}

/** Lay nodes out in dependency layers: prerequisites left, dependents right. */
function layoutGraph(graph: ConceptGraph): {
  positioned: PositionedNode[]
  edges: PositionedEdge[]
  width: number
  height: number
} {
  const nodes = graph.concepts
  if (nodes.length === 0) {
    return { positioned: [], edges: [], width: 1000, height: 0 }
  }

  // Depth = longest prerequisite chain ending at a node (layer index).
  const byName = new Map(nodes.map((n) => [n.concept, n]))
  const incoming = new Map<string, string[]>()
  nodes.forEach((n) => incoming.set(n.concept, []))
  graph.edges.forEach((e) => {
    if (byName.has(e.prerequisite) && byName.has(e.concept)) {
      incoming.get(e.concept)!.push(e.prerequisite)
    }
  })

  const depthCache = new Map<string, number>()
  const depthOf = (name: string, seen: Set<string>): number => {
    if (depthCache.has(name)) return depthCache.get(name)!
    if (seen.has(name)) return 0 // cycle guard
    seen.add(name)
    const preds = incoming.get(name) ?? []
    const d = preds.length === 0 ? 0 : 1 + Math.max(...preds.map((p) => depthOf(p, seen)))
    seen.delete(name)
    depthCache.set(name, d)
    return d
  }

  const layers = new Map<number, GraphNode[]>()
  nodes.forEach((n) => {
    const d = depthOf(n.concept, new Set())
    if (!layers.has(d)) layers.set(d, [])
    layers.get(d)!.push(n)
  })

  const maxDepth = Math.max(0, ...[...layers.keys()])
  const W = 1000
  const colGap = maxDepth === 0 ? 0 : W / (maxDepth + 1)
  const rowGap = 92
  const positioned: PositionedNode[] = []
  const posByName = new Map<string, { x: number; y: number }>()

  for (let d = 0; d <= maxDepth; d++) {
    const layer = layers.get(d) ?? []
    const colX = maxDepth === 0 ? W / 2 : colGap * d + colGap / 2
    layer.forEach((node, i) => {
      const y = (i + 1) * rowGap
      positioned.push({ node, x: colX, y })
      posByName.set(node.concept, { x: colX, y })
    })
  }

  const edges = graph.edges
    .map((e) => {
      const from = posByName.get(e.prerequisite)
      const to = posByName.get(e.concept)
      if (!from || !to) return null
      return { from, to, prerequisite: e.prerequisite, concept: e.concept }
    })
    .filter((e): e is PositionedEdge => e !== null)

  const maxRows = Math.max(1, ...[...layers.values()].map((l) => l.length))
  return { positioned, edges, width: W, height: (maxRows + 1) * rowGap }
}

const LEGEND = [
  { color: scoreTone(100).stroke, label: 'Strong' },
  { color: scoreTone(50).stroke, label: 'Shaky' },
  { color: scoreTone(0).stroke, label: 'Weak' },
]

interface ConceptGraphMapProps {
  graph: ConceptGraph | null
  isError: boolean
  onRetry: () => void
}

/**
 * Concept prerequisite graph — layered SVG node/edge map on paper. Mastery is
 * colored via the app-wide semantic scoreTone. Product differentiator: wide,
 * prominent tile with a legend and "fix the foundation first" blockers.
 */
export function ConceptGraphMap({ graph, isError, onRetry }: ConceptGraphMapProps) {
  const layout = useMemo(() => (graph ? layoutGraph(graph) : null), [graph])
  const hasData = !!graph && graph.concepts.length > 0

  return (
    <Card accent padding="lg" elevation={2}>
      <div className="flex items-start justify-between gap-3">
        <SectionHead
          num="02"
          title="Concept map"
          hint="How concepts build on each other — fix the upstream gaps first"
          className="flex-1"
        />
        {hasData && (
          <div className="hidden sm:flex items-center gap-3 flex-shrink-0 pt-1">
            {LEGEND.map((it) => (
              <span key={it.label} className="inline-flex items-center gap-1.5 text-[11px] text-ink-soft">
                <span className="w-2.5 h-2.5 rounded-full" style={{ background: it.color }} />
                {it.label}
              </span>
            ))}
          </div>
        )}
      </div>

      {isError ? (
        <ErrorState compact title="Couldn't load your concept map." onRetry={onRetry} />
      ) : !hasData || !layout || layout.positioned.length === 0 ? (
        <EmptyState
          icon={<Network />}
          title="Concept map not built yet"
          description="Study a few topics — we’ll map how your concepts depend on each other."
        />
      ) : (
        <div className="overflow-x-auto -mx-2 px-2">
          <svg
            viewBox={`0 0 ${layout.width} ${layout.height}`}
            className="w-full min-w-[640px]"
            style={{ height: Math.min(420, layout.height) }}
          >
            {/* Edges */}
            {layout.edges.map((e, i) => {
              const midX = (e.from.x + e.to.x) / 2
              const d = `M ${e.from.x} ${e.from.y} C ${midX} ${e.from.y}, ${midX} ${e.to.y}, ${e.to.x} ${e.to.y}`
              return (
                <motion.path
                  key={`${e.prerequisite}->${e.concept}`}
                  d={d}
                  fill="none"
                  stroke={EDGE_COLOR}
                  strokeWidth={1.5}
                  initial={{ pathLength: 0, opacity: 0 }}
                  animate={{ pathLength: 1, opacity: 1 }}
                  transition={{ duration: 0.35, delay: Math.min(0.1 + i * 0.02, 0.3), ease: 'easeOut' }}
                />
              )
            })}
            {/* Nodes */}
            {layout.positioned.map(({ node, x, y }, i) => {
              const color = node.has_data ? scoreTone(node.mastery_pct).stroke : NO_DATA_COLOR
              const label = node.concept.length > 22 ? node.concept.slice(0, 21) + '…' : node.concept
              const w = Math.max(120, label.length * 7.6 + 28)
              return (
                <motion.g
                  key={node.concept}
                  initial={{ opacity: 0, scale: 0.85 }}
                  animate={{ opacity: 1, scale: 1 }}
                  transition={{ duration: 0.3, delay: Math.min(0.05 + i * 0.03, 0.3), ease: [0.22, 1, 0.36, 1] }}
                >
                  <rect
                    x={x - w / 2}
                    y={y - 17}
                    width={w}
                    height={34}
                    rx={9}
                    fill={NODE_FILL}
                    stroke={color}
                    strokeWidth={1.5}
                    style={{ filter: 'drop-shadow(0 1px 2px rgba(33,31,26,0.08))' }}
                  />
                  <circle cx={x - w / 2 + 14} cy={y} r={4} fill={color} />
                  <text x={x - w / 2 + 26} y={y + 4} fontSize={12} fill={LABEL_COLOR} fontWeight={500}>
                    {label}
                  </text>
                  {node.has_data && (
                    <text x={x + w / 2 - 10} y={y + 4} fontSize={11} fill={color} textAnchor="end" fontWeight={600}>
                      {Math.round(node.mastery_pct)}%
                    </text>
                  )}
                </motion.g>
              )
            })}
          </svg>
        </div>
      )}

      {/* Foundation-first blockers, surfaced under the graph they explain. */}
      {!isError && graph && graph.blockers.length > 0 && (
        <div className="mt-5 pt-4 border-t border-line">
          <p className="text-xs font-semibold text-warning mb-3">Fix the foundation first</p>
          <div className="space-y-2">
            {graph.blockers.slice(0, MAX_BLOCKERS).map((b) => (
              <div key={`${b.prerequisite}->${b.concept}`} className="flex items-center flex-wrap gap-2 text-sm">
                <span className="text-warning bg-warning-wash border border-warning/25 rounded px-2 py-0.5 text-xs">
                  {b.prerequisite} <span className="text-warning/70 tnum">({Math.round(b.prerequisite_pct)}%)</span>
                </span>
                <ArrowRight className="w-3.5 h-3.5 text-ink-faint" />
                <span className="text-ink-soft text-xs">
                  {b.concept} <span className="text-ink-faint tnum">({Math.round(b.concept_pct)}%)</span>
                </span>
              </div>
            ))}
          </div>
        </div>
      )}
    </Card>
  )
}
