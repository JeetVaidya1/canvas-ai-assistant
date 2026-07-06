import type { ReactNode } from 'react'
import { Link } from 'react-router-dom'
import { BrandMark } from '@/components/ui/BrandMark'

const CONTACT_EMAIL = 'hello@vindexa.app'

function FooterColumn({ title, children }: { title: string; children: ReactNode }) {
  return (
    <div>
      <p className="text-[10px] font-semibold uppercase tracking-[0.18em] text-ink-faint mb-3">{title}</p>
      <ul className="space-y-2">{children}</ul>
    </div>
  )
}

function FooterLink({ to, children }: { to: string; children: ReactNode }) {
  return (
    <li>
      <Link to={to} className="text-sm text-ink-soft hover:text-ink transition-colors focus-ring rounded">
        {children}
      </Link>
    </li>
  )
}

/**
 * Shared footer for all public pages: hairline top, three quiet columns
 * (Product / Company / Legal), brand line underneath.
 */
export function PublicFooter() {
  return (
    <footer className="border-t border-line mt-8">
      <div className="max-w-6xl mx-auto px-6 py-12">
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-10">
          <div className="col-span-2 sm:col-span-1">
            <div className="flex items-center gap-2 mb-3">
              <BrandMark className="w-6 h-6" />
              <span className="font-display text-base font-semibold text-ink tracking-tight">Vindexa</span>
            </div>
            <p className="text-xs text-ink-faint leading-relaxed max-w-[26ch]">
              Grounded in your materials. Nothing else.
            </p>
          </div>
          <FooterColumn title="Product">
            <FooterLink to="/pricing">Pricing</FooterLink>
            <FooterLink to="/help">Help</FooterLink>
            <FooterLink to="/login">Sign in</FooterLink>
          </FooterColumn>
          <FooterColumn title="Company">
            <li className="text-sm text-ink-soft">A study system, not a shortcut.</li>
            <li>
              <a
                href={`mailto:${CONTACT_EMAIL}`}
                className="text-sm text-ink-soft hover:text-ink transition-colors focus-ring rounded"
              >
                {CONTACT_EMAIL}
              </a>
            </li>
          </FooterColumn>
          <FooterColumn title="Legal">
            <FooterLink to="/terms">Terms of Service</FooterLink>
            <FooterLink to="/privacy">Privacy Policy</FooterLink>
          </FooterColumn>
        </div>
        <div className="border-t border-line mt-10 pt-6 flex flex-col sm:flex-row items-center justify-between gap-2">
          <span className="text-xs text-ink-faint">© {new Date().getFullYear()} Vindexa. All rights reserved.</span>
          <span className="text-xs text-ink-faint">Your files stay yours — scoped to your account.</span>
        </div>
      </div>
    </footer>
  )
}
