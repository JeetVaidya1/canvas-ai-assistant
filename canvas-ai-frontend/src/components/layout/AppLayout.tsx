import { Outlet, useLocation } from 'react-router-dom'
import { AnimatePresence, motion } from 'motion/react'
import AppSidebar from './AppSidebar'
import TopBar from './TopBar'

export default function AppLayout() {
  const location = useLocation()

  return (
    <div className="flex h-screen text-zinc-50">
      <AppSidebar />
      <div className="flex flex-col flex-1 min-w-0">
        <TopBar />
        <main className="flex-1 overflow-y-auto">
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
    </div>
  )
}
