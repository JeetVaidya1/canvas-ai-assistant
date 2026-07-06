import { useNavigate } from 'react-router-dom'
import { Check, ArrowRight } from 'lucide-react'
import { Button } from '@/components/ui/Button'
import { Badge } from '@/components/ui/Badge'
import { PublicNav } from '@/components/marketing/PublicNav'
import { PublicFooter } from '@/components/marketing/PublicFooter'
import { SectionHead } from '@/components/marketing/SectionHead'

const PLAN_FEATURES = [
  'Course chat with page-level citations from your own files',
  'Socratic & Feynman tutoring modes',
  'Adaptive quiz drills and practice problems',
  'Timed mock exams with partial credit and per-concept grading',
  'Spaced review queue built from your mistakes',
  'Notes and flashcards, exportable to PDF and Anki',
  'Per-topic exam readiness, updated as you study',
  'PDF, DOCX and PPTX uploads + Canvas LMS import',
]

const PRICING_FAQS = [
  {
    question: 'How does the free trial work?',
    answer:
      '7 days, full access, no credit card required to start. If Vindexa isn’t helping by the end of the week, walk away — nothing to cancel, nothing charged.',
  },
  {
    question: 'What happens when my trial ends?',
    answer:
      'Your courses and files stay stored and scoped to your account. The study tools lock until you subscribe — nothing is deleted, and you pick up exactly where you left off.',
  },
  {
    question: 'Is there a free tier?',
    answer:
      'No. There’s one paid plan and a real trial instead of a crippled free version. That keeps the product honest: every feature you try during the trial is the actual product.',
  },
  {
    question: 'Can I cancel anytime?',
    answer:
      'Yes. Subscriptions are month-to-month with no lock-in. Cancel whenever and keep access through the end of the period you paid for.',
  },
]

export default function PricingPage() {
  const navigate = useNavigate()
  const toLogin = () => navigate('/login')

  return (
    <div className="relative w-full min-h-screen bg-paper text-ink">
      <PublicNav />

      {/* ===== HEADLINE ===== */}
      <section className="px-6 pt-28 pb-14 sm:pt-36">
        <div className="max-w-2xl mx-auto text-center animate-fade-up">
          <h1 className="font-display text-4xl sm:text-5xl font-semibold tracking-tight leading-[1.1] text-ink">
            One plan. <span className="hl">Everything included.</span>
          </h1>
          <p className="mt-5 text-lg text-ink-soft leading-relaxed">
            No feature tiers, no usage anxiety. Try the whole product free for a week,
            then one flat price for every course you’re taking.
          </p>
        </div>
      </section>

      {/* ===== PLAN CARD ===== */}
      <section className="px-6 pb-20">
        <div className="max-w-xl mx-auto animate-fade-up">
          <div className="rounded-2xl card-surface elev-2 p-8 sm:p-10">
            <div className="flex items-start justify-between gap-4 mb-6">
              <div>
                <h2 className="font-display text-2xl font-semibold text-ink tracking-tight">Vindexa</h2>
                <p className="text-sm text-ink-soft mt-1">Every feature, every course.</p>
              </div>
              <Badge tone="marker">Draft pricing</Badge>
            </div>

            <div className="flex items-baseline gap-2 mb-1">
              <span className="font-display text-5xl font-semibold text-ink tnum">$15</span>
              <span className="text-sm text-ink-soft">/ month</span>
            </div>
            <p className="text-sm text-ink-soft mb-8">7-day free trial · no credit card required</p>

            <ul className="space-y-3 mb-9">
              {PLAN_FEATURES.map((feature) => (
                <li key={feature} className="flex items-start gap-3">
                  <Check className="w-4 h-4 text-success mt-0.5 flex-shrink-0" />
                  <span className="text-sm text-ink leading-relaxed">{feature}</span>
                </li>
              ))}
            </ul>

            <Button size="lg" className="w-full" onClick={toLogin} rightIcon={<ArrowRight className="w-4 h-4" />}>
              Start free trial
            </Button>
            <p className="text-xs text-ink-faint text-center mt-4">
              Pricing may change before launch. Trial terms shown are the current plan of record.
            </p>
          </div>
        </div>
      </section>

      {/* ===== PRICING FAQ ===== */}
      <section className="px-6 pb-24">
        <div className="max-w-3xl mx-auto">
          <SectionHead num="01" title="Pricing questions" />
          <dl>
            {PRICING_FAQS.map((faq) => (
              <div key={faq.question} className="border-t border-line py-6 grid grid-cols-1 md:grid-cols-[220px_1fr] gap-2 md:gap-8">
                <dt className="text-sm font-semibold text-ink leading-snug">{faq.question}</dt>
                <dd className="text-sm text-ink-soft leading-relaxed">{faq.answer}</dd>
              </div>
            ))}
          </dl>
        </div>
      </section>

      <PublicFooter />
    </div>
  )
}
