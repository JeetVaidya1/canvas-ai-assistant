import { useState } from 'react'
import { AnimatePresence, motion } from 'motion/react'
import { ChevronDown, FileText } from 'lucide-react'
import { cn } from '@/lib/utils'
import { CitationChip } from '@/components/learn/CitationChip'
import type { Source } from '@/lib/api'

interface SourcesDisclosureProps {
  sources: ReadonlyArray<Source>
}

/** Collapsible source citations — tidy "N sources" pill that expands to cited pages. */
export function SourcesDisclosure({ sources }: SourcesDisclosureProps) {
  const [open, setOpen] = useState(false)
  if (sources.length === 0) return null
  return (
    <div className="mt-3">
      <button
        onClick={() => setOpen((v) => !v)}
        className="inline-flex items-center gap-1.5 rounded-full border border-accent-line bg-accent-wash px-2.5 py-1 text-[11px] font-medium text-accent-deep transition-colors hover:border-accent/50"
      >
        <FileText className="h-3 w-3" />
        {sources.length} source{sources.length !== 1 ? 's' : ''} from your materials
        <ChevronDown className={cn('h-3 w-3 transition-transform', open && 'rotate-180')} />
      </button>
      <AnimatePresence initial={false}>
        {open && (
          <motion.div
            initial={{ opacity: 0, height: 0 }}
            animate={{ opacity: 1, height: 'auto' }}
            exit={{ opacity: 0, height: 0 }}
            transition={{ duration: 0.2 }}
            className="overflow-hidden"
          >
            <div className="mt-2 flex flex-wrap gap-1.5">
              {sources.map((s, i) => (
                <CitationChip key={`${s.file}-${s.page ?? 'x'}-${i}`} file={s.file} page={s.page} />
              ))}
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  )
}
