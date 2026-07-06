import { useNavigate } from 'react-router-dom'
import { LogOut, Settings } from 'lucide-react'
import { useAuth } from '@/lib/auth'
import { useProfile } from '@/hooks/useProfile'
import { usePopover } from './usePopover'

/** TopBar avatar: ink disc with the user's initial; popover with account actions. */
export default function AvatarMenu() {
  const navigate = useNavigate()
  const { user, signOut } = useAuth()
  const { displayName } = useProfile()
  const { open, setOpen, ref } = usePopover<HTMLDivElement>()

  const email = user?.email ?? ''
  const initial = (displayName || email || 'U').charAt(0).toUpperCase()

  const itemClass =
    'w-full flex items-center gap-2 px-3 py-1.5 text-left text-[13px] font-medium text-ink-soft hover:text-ink hover:bg-line/40 transition-colors'

  return (
    <div ref={ref} className="relative">
      <button
        onClick={() => setOpen(!open)}
        aria-haspopup="menu"
        aria-expanded={open}
        aria-label="Account menu"
        className="w-7 h-7 rounded-full bg-ink flex items-center justify-center text-[11px] font-semibold text-paper focus-ring"
      >
        {initial}
      </button>

      {open && (
        <div
          role="menu"
          className="absolute right-0 top-full mt-1.5 w-56 py-1.5 bg-surface border border-line rounded-lg elev-3 z-50"
        >
          <div className="px-3 pt-1 pb-2 border-b border-line">
            {displayName && <div className="text-[13px] font-semibold text-ink truncate">{displayName}</div>}
            <div className="text-xs text-ink-faint truncate">{email || 'Signed in'}</div>
          </div>
          <div className="pt-1.5">
            <button
              role="menuitem"
              onClick={() => {
                setOpen(false)
                navigate('/settings')
              }}
              className={itemClass}
            >
              <Settings className="w-3.5 h-3.5 flex-shrink-0" />
              Settings
            </button>
            <button
              role="menuitem"
              onClick={() => {
                setOpen(false)
                void signOut()
              }}
              className={itemClass}
            >
              <LogOut className="w-3.5 h-3.5 flex-shrink-0" />
              Sign out
            </button>
          </div>
        </div>
      )}
    </div>
  )
}
