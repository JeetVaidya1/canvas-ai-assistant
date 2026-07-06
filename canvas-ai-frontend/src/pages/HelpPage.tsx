import type { ReactNode } from 'react'
import { useNavigate } from 'react-router-dom'
import { ArrowRight } from 'lucide-react'
import { Button } from '@/components/ui/Button'
import { PublicNav } from '@/components/marketing/PublicNav'
import { PublicFooter } from '@/components/marketing/PublicFooter'

/** Inline chip for a UI term as it appears in the product ("Course Brief", "Today"). */
function UITerm({ children }: { children: ReactNode }) {
  return (
    <span className="inline-block rounded border border-line bg-paper-deep px-1.5 py-px text-[0.85em] font-medium text-ink align-baseline">
      {children}
    </span>
  )
}

interface HelpSectionProps {
  num: string
  title: string
  children: ReactNode
}

/** Numbered doc section — mirrors the syllabus treatment at article scale. */
function HelpSection({ num, title, children }: HelpSectionProps) {
  return (
    <section className="border-t border-line pt-8">
      <div className="section-head mb-5">
        <span className="section-num">{num}</span>
        <h2 className="font-display text-xl md:text-2xl font-semibold text-ink tracking-tight">{title}</h2>
      </div>
      <div className="space-y-3 text-sm text-ink-soft leading-relaxed">{children}</div>
    </section>
  )
}

export default function HelpPage() {
  const navigate = useNavigate()

  return (
    <div className="relative w-full min-h-screen bg-paper text-ink">
      <PublicNav />

      <main className="px-6 pt-28 pb-20 sm:pt-32">
        <article className="max-w-2xl mx-auto animate-fade-up">
          <h1 className="font-display text-3xl sm:text-4xl font-semibold tracking-tight text-ink mb-3">
            Getting started
          </h1>
          <p className="text-base text-ink-soft leading-relaxed mb-12">
            The whole loop in one page: add a course, feed it your materials, and let the system
            tell you what to study next. Ten minutes of setup, then it compounds.
          </p>

          <div className="space-y-12">
            <HelpSection num="01" title="Create a course">
              <p>
                After signing in, create a course from the dashboard — one course per class you&rsquo;re
                taking. Each course is its own sealed workspace: its files, chat history, practice
                record and readiness score never mix with your other courses.
              </p>
            </HelpSection>

            <HelpSection num="02" title="Add your materials">
              <p>
                Open <UITerm>Materials</UITerm> and upload your slides, readings and notes as PDF,
                DOCX or PPTX. Or connect <UITerm>Canvas LMS</UITerm> to import course files, the
                syllabus and exam dates in one step.
              </p>
              <p>
                Everything you add is indexed — chunked, embedded and cross-linked — so the rest of
                the product can retrieve from it. More material means better answers and better
                practice; the syllabus is especially valuable because it tells the system what your
                exam actually covers.
              </p>
            </HelpSection>

            <HelpSection num="03" title="Read your Course Brief">
              <p>
                Once your files are indexed, the <UITerm>Course Brief</UITerm> shows the topic map
                Vindexa extracted from your materials: each topic, with the documents and page ranges
                that cover it (for example, <span className="font-mono text-xs">Graphs.pdf · pp. 3–18</span>).
              </p>
              <p>
                This is your sanity check. If a topic you know is on the exam isn&rsquo;t in the
                Brief, the system doesn&rsquo;t have material for it yet — upload the missing lecture
                or reading and rebuild the Brief.
              </p>
            </HelpSection>

            <HelpSection num="04" title="Learn — and check the citations">
              <p>
                Ask anything in <UITerm>Learn</UITerm>. Every answer is retrieved from your own files
                and carries citation chips naming the exact file and pages it came from — tap one to
                see the source. If your materials don&rsquo;t cover a question, Vindexa says so rather
                than improvising.
              </p>
              <p>
                Prefer being led to the answer? Switch to <UITerm>Socratic</UITerm> mode to be guided
                by questions, or <UITerm>Feynman</UITerm> mode to explain a concept back and have the
                gaps in your explanation pinpointed.
              </p>
            </HelpSection>

            <HelpSection num="05" title="Drill, and rate your confidence">
              <p>
                <UITerm>Practice</UITerm> generates quiz drills and problem sets from your materials.
                After picking an answer you can tag it <UITerm>Sure</UITerm>, <UITerm>Think so</UITerm> or{' '}
                <UITerm>Guessing</UITerm> — this calibration matters: a confident wrong answer is a
                bigger red flag than an honest guess, and the system weights it accordingly.
              </p>
              <p>
                Everything you miss enters the spaced review queue, scheduled to resurface right
                before you&rsquo;d forget it. When you&rsquo;re closer to the exam,{' '}
                <UITerm>Exam</UITerm> runs full timed simulations graded with partial credit and a
                per-concept breakdown.
              </p>
            </HelpSection>

            <HelpSection num="06" title="Follow Readiness and Today">
              <p>
                Your course home leads with two things. The <UITerm>Readiness</UITerm> score
                aggregates your per-topic performance — drills, practice, exams, recency — into one
                number that moves as you study. And the <UITerm>Today</UITerm> panel turns that into a
                short, concrete plan: which topics are weakest, what to review, what to drill next.
              </p>
              <p>
                That&rsquo;s the loop. You don&rsquo;t decide what to study from a vague sense of
                dread — you open the course and the next most useful thing is already at the top.
              </p>
            </HelpSection>
          </div>

          {/* Bottom CTA */}
          <div className="border-t border-line mt-12 pt-10 text-center">
            <p className="text-sm text-ink-soft mb-5">
              Setup to first cited answer takes about two minutes.
            </p>
            <Button size="lg" onClick={() => navigate('/login')} rightIcon={<ArrowRight className="w-4 h-4" />}>
              Start free
            </Button>
          </div>
        </article>
      </main>

      <PublicFooter />
    </div>
  )
}
