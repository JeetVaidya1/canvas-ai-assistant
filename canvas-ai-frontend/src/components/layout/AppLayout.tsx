import { useCallback, useEffect, useState } from 'react'
import { Outlet, useLocation } from 'react-router-dom'
import { AnimatePresence, motion } from 'motion/react'
import TopBar from '@/components/shell/TopBar'
import IconRail, { MobileNavBar } from '@/components/shell/IconRail'
import CommandPalette from '@/components/shell/CommandPalette'
import CreateCourseModal from '@/components/CreateCourseModal'

/**
 * Command-workspace shell: slim TopBar (brand, course switcher, Cmd+K,
 * readiness, avatar) over an icon rail + single scroll container. Below md
 * the rail becomes a bottom icon bar (see MobileNavBar).
 */
export default function AppLayout() {
  const location = useLocation()
  const [paletteOpen, setPaletteOpen] = useState(false)
  const [createOpen, setCreateOpen] = useState(false)

  const openPalette = useCallback(() => setPaletteOpen(true), [])
  const closePalette = useCallback(() => setPaletteOpen(false), [])
  const openCreate = useCallback(() => setCreateOpen(true), [])

  // Global Cmd+K / Ctrl+K toggle.
  useEffect(() => {
    const onKeyDown = (e: KeyboardEvent) => {
      if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === 'k') {
        e.preventDefault()
        setPaletteOpen((open) => !open)
      }
    }
    window.addEventListener('keydown', onKeyDown)
    return () => window.removeEventListener('keydown', onKeyDown)
  }, [])

  return (
    <div className="flex flex-col h-screen text-ink bg-paper">
      <TopBar onOpenPalette={openPalette} onNewCourse={openCreate} />
      <div className="flex flex-1 min-h-0">
        <IconRail />
        {/* The single app scroll container — pages either scroll here or opt
            into their own internal layout with h-full. No backdrop-blur
            wrappers around the outlet: blur over a scrolling transcript is a
            compositor hazard. */}
        <main className="relative flex-1 min-w-0 min-h-0 overflow-y-auto">
          <AnimatePresence mode="wait">
            <motion.div
              key={location.pathname}
              initial={{ opacity: 0, y: 4 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0 }}
              transition={{ duration: 0.15, ease: 'easeOut' }}
              className="h-full"
            >
              <Outlet />
            </motion.div>
          </AnimatePresence>
        </main>
      </div>
      <MobileNavBar />
      <CommandPalette open={paletteOpen} onClose={closePalette} onNewCourse={openCreate} />
      <CreateCourseModal open={createOpen} onClose={() => setCreateOpen(false)} />
    </div>
  )
}
