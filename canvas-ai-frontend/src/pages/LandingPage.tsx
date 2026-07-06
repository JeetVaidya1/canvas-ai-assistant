import { Link, useNavigate } from 'react-router-dom'
import {
  MessageCircle,
  Target,
  ClipboardList,
  RefreshCw,
  FileText,
  Upload,
  Layers,
  Timer,
  CheckCircle,
  ArrowRight,
  BookOpen,
} from 'lucide-react'
import { Button } from '@/components/ui/Button'
import { Badge } from '@/components/ui/Badge'
import { ProgressRing } from '@/components/ui/Progress'
import { PublicNav } from '@/components/marketing/PublicNav'
import { PublicFooter } from '@/components/marketing/PublicFooter'
import { SectionHead } from '@/components/marketing/SectionHead'

const FEATURES = [
  {
    icon: MessageCircle,
    title: 'Answers with page citations',
    description: 'Ask anything about your course. Every answer is retrieved from your own files and cites the exact source — no internet guesswork.',
  },
  {
    icon: BookOpen,
    title: 'Socratic & Feynman tutoring',
    description: 'Get guided to the answer with questions, or explain a concept back and have the gaps in your explanation pinpointed.',
  },
  {
    icon: Target,
    title: 'Practice that adapts',
    description: 'Quiz drills and problem sets that re-weight toward the topics you keep missing, at the difficulty you need.',
  },
  {
    icon: ClipboardList,
    title: 'Timed mock exams',
    description: 'Full exam simulations built from your materials, graded with partial credit and per-concept breakdowns.',
  },
  {
    icon: RefreshCw,
    title: 'Spaced review queue',
    description: 'Every mistake is scheduled for review right before you’d forget it. Nothing you got wrong slips away.',
  },
  {
    icon: FileText,
    title: 'Notes & flashcards',
    description: 'Condense any topic into structured notes with auto-generated flashcards, exportable to PDF and Anki.',
  },
]

const STEPS = [
  {
    icon: Upload,
    number: '01',
    title: 'Add your course',
    description: 'Upload slides, PDFs and docs — or import straight from Canvas LMS, syllabus and exam dates included.',
  },
  {
    icon: Layers,
    number: '02',
    title: 'Vindexa indexes it',
    description: 'Your materials are chunked, embedded and cross-linked into a course-specific knowledge base.',
  },
  {
    icon: Timer,
    number: '03',
    title: 'Study with direction',
    description: 'Chat, drill, simulate exams — while your readiness score tells you exactly what to study next.',
  },
]

const BENEFITS = [
  'Every answer traceable to your own materials',
  'Wrong answers automatically reshape your practice',
  'Exam readiness scored per topic, updated as you study',
  'Works with PDF, DOCX, PPTX and Canvas LMS',
  'Your files stay yours — scoped to your account',
]

const FAQS = [
  {
    question: 'How does grounding actually work?',
    answer:
      'Every answer is retrieved from the files you upload — your slides, readings and notes are chunked, embedded and searched per question, and each response cites the exact file and page it came from. If your materials don’t cover something, Vindexa says so instead of guessing from the open internet.',
  },
  {
    question: 'What file types work? Does Canvas import work?',
    answer:
      'PDF, DOCX and PPTX uploads are supported. You can also connect Canvas LMS to import course files, the syllabus and exam dates directly, so you don’t have to download and re-upload everything by hand.',
  },
  {
    question: 'How is my readiness score computed?',
    answer:
      'Readiness is calculated per topic from your actual performance — quiz drills, practice problems, mock exams and how recently you got each topic right. It updates as you study. It’s a signal for what to work on next, not a prediction of your grade.',
  },
  {
    question: 'Is my data private?',
    answer:
      'Your files are stored in Supabase, scoped to your account, and used only to power your own study tools. Course content is processed via the Anthropic API to generate answers. We never sell your data, and nothing you upload is shared with other users.',
  },
  {
    question: 'What does it cost?',
    answer:
      'One plan, around $15/month (draft pricing), with a 7-day free trial and no credit card required to start. There’s no free-forever tier — the trial is the free part.',
  },
  {
    question: 'Does this replace studying?',
    answer:
      'No — and it isn’t meant to. Vindexa directs your effort: it finds your weak topics, drills them, and schedules review before you forget. You still do the reading, answer the questions, and sit the mock exams. It makes the work count; it doesn’t do the work.',
  },
]

/** DOM-built product preview — a white sheet with paper window chrome. */
function ProductPreview() {
  return (
    <div className="relative mx-auto max-w-4xl rounded-2xl border border-line bg-surface elev-3 overflow-hidden text-left">
      {/* Window chrome */}
      <div className="flex items-center gap-1.5 px-4 h-9 border-b border-line bg-paper-deep">
        <span className="w-2.5 h-2.5 rounded-full bg-line-strong" />
        <span className="w-2.5 h-2.5 rounded-full bg-line-strong" />
        <span className="w-2.5 h-2.5 rounded-full bg-line-strong" />
        <span className="ml-3 text-[11px] text-ink-faint truncate">Data Structures · Learn</span>
      </div>
      <div className="grid grid-cols-1 md:grid-cols-[1fr_220px]">
        {/* Chat pane */}
        <div className="p-5 sm:p-6 space-y-4 min-w-0">
          <div className="flex justify-end">
            <div className="rounded-xl rounded-br-sm bg-paper-deep border border-line px-3.5 py-2 text-sm text-ink max-w-[85%]">
              When should I use Dijkstra instead of A*?
            </div>
          </div>
          <div className="max-w-[92%]">
            <p className="text-sm text-ink-soft leading-relaxed">
              Use Dijkstra when you have no admissible heuristic or need shortest paths to
              <span className="text-ink font-medium"> every</span> node; A* wins when a good heuristic
              (like straight-line distance) can steer the search toward a single goal…
            </p>
            <div className="flex flex-wrap items-center gap-2 mt-3">
              <Badge tone="accent" icon={<BookOpen />}>Lecture 8 · pp. 12–14</Badge>
              <Badge tone="neutral">Graphs.pdf · p. 3</Badge>
            </div>
          </div>
          <div className="flex flex-wrap gap-2 pt-1">
            {['Quiz me on this', 'Show a worked example'].map((chip) => (
              <span key={chip} className="text-xs text-ink-faint border border-line rounded-full px-3 py-1">
                {chip}
              </span>
            ))}
          </div>
        </div>
        {/* Readiness rail */}
        <div className="border-t md:border-t-0 md:border-l border-line p-5 bg-paper">
          <p className="text-[10px] font-semibold uppercase tracking-[0.18em] text-ink-faint mb-3">Exam readiness</p>
          <div className="flex md:flex-col items-center md:items-start gap-4">
            <ProgressRing value={72} size={84} strokeWidth={7}>
              <span className="text-lg font-semibold text-success tnum">72%</span>
            </ProgressRing>
            <div className="space-y-1.5">
              <p className="text-[10px] font-semibold uppercase tracking-[0.18em] text-ink-faint">Focus next</p>
              <div className="flex flex-wrap gap-1.5">
                <Badge tone="warning">AVL rotations</Badge>
                <Badge tone="warning">Graph traversal</Badge>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}

export default function LandingPage() {
  const navigate = useNavigate()
  const toLogin = () => navigate('/login')

  return (
    <div className="relative w-full min-h-screen bg-paper text-ink">
      <PublicNav />

      {/* ===== HERO ===== */}
      <section className="px-6 pt-32 pb-20 sm:pt-40">
        <div className="max-w-3xl mx-auto text-center animate-fade-up">
          <h1 className="font-display text-4xl sm:text-5xl md:text-6xl font-semibold tracking-tight leading-[1.08] text-ink">
            Your course materials,
            <br />
            turned into a <span className="hl">study system</span>
          </h1>
          <p className="mt-6 text-lg text-ink-soft leading-relaxed max-w-2xl mx-auto">
            Vindexa indexes your slides, readings and syllabus — then answers with page-level
            citations, drills your weak topics, and tells you when you’re ready for the exam.
          </p>
          <div className="flex flex-col sm:flex-row gap-3 justify-center mt-9">
            <Button size="lg" onClick={toLogin} rightIcon={<ArrowRight className="w-4 h-4" />}>
              Start free
            </Button>
            <Button
              size="lg"
              variant="secondary"
              onClick={() => document.getElementById('how-it-works')?.scrollIntoView({ behavior: 'smooth' })}
            >
              See how it works
            </Button>
          </div>
        </div>
        <div className="mt-16 px-0 sm:px-6 animate-fade-up">
          <ProductPreview />
        </div>
      </section>

      {/* ===== HOW IT WORKS ===== */}
      <section id="how-it-works" className="py-20 px-6">
        <div className="max-w-5xl mx-auto">
          <SectionHead num="01" title="From file dump to study plan" />
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            {STEPS.map((step) => (
              <div key={step.number} className="relative p-6 rounded-xl card-surface">
                <span className="font-mono text-[2.75rem] text-line-strong absolute top-3 right-5 leading-none select-none">
                  {step.number}
                </span>
                <div className="w-10 h-10 rounded-lg bg-paper-deep border border-line flex items-center justify-center mb-5">
                  <step.icon className="w-5 h-5 text-ink-soft" />
                </div>
                <h3 className="text-base font-semibold text-ink mb-2">{step.title}</h3>
                <p className="text-sm text-ink-soft leading-relaxed">{step.description}</p>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* ===== FEATURES ===== */}
      <section className="py-20 px-6">
        <div className="max-w-5xl mx-auto">
          <SectionHead num="02" title="One connected system" />
          <p className="text-ink-soft text-base max-w-lg -mt-6 mb-10">
            Every tool feeds the next: chat surfaces gaps, practice targets them, review locks them in.
          </p>
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
            {FEATURES.map((feature) => (
              <div key={feature.title} className="group p-6 rounded-xl card-surface card-interactive">
                <feature.icon className="w-5 h-5 text-ink-soft group-hover:text-accent transition-colors mb-4" />
                <h3 className="text-sm font-semibold text-ink mb-2">{feature.title}</h3>
                <p className="text-sm text-ink-soft leading-relaxed">{feature.description}</p>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* ===== WHY ===== */}
      <section className="py-20 px-6">
        <div className="max-w-5xl mx-auto">
          <SectionHead num="03" title="Built for how students actually study" />
          <div className="grid grid-cols-1 md:grid-cols-2 gap-12 items-center">
            <p className="text-ink-soft text-sm leading-relaxed">
              Generic AI tools answer from the open internet and forget you between sessions.
              Vindexa is grounded in your specific course and remembers every answer you get
              wrong — so the system always knows your weakest topic, and so do you.
            </p>
            <div className="space-y-3">
              {BENEFITS.map((benefit) => (
                <div key={benefit} className="flex items-start gap-3 p-3.5 rounded-xl card-surface">
                  <CheckCircle className="w-4 h-4 text-success mt-0.5 flex-shrink-0" />
                  <span className="text-sm text-ink">{benefit}</span>
                </div>
              ))}
            </div>
          </div>
        </div>
      </section>

      {/* ===== FAQ ===== */}
      <section className="py-20 px-6">
        <div className="max-w-3xl mx-auto">
          <SectionHead num="04" title="Questions" />
          <dl>
            {FAQS.map((faq) => (
              <div key={faq.question} className="border-t border-line py-6 grid grid-cols-1 md:grid-cols-[220px_1fr] gap-2 md:gap-8">
                <dt className="text-sm font-semibold text-ink leading-snug">{faq.question}</dt>
                <dd className="text-sm text-ink-soft leading-relaxed">{faq.answer}</dd>
              </div>
            ))}
          </dl>

          {/* Pricing teaser */}
          <div className="border-t border-line mt-0 pt-8 flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4">
            <div>
              <p className="font-display text-xl font-semibold text-ink tracking-tight">
                One plan. <span className="tnum">$15</span>/month.
              </p>
              <p className="text-sm text-ink-soft mt-1">7-day free trial, no credit card. Draft pricing — may change before launch.</p>
            </div>
            <Link
              to="/pricing"
              className="inline-flex items-center gap-1.5 text-sm font-medium text-accent hover:text-accent-deep transition-colors focus-ring rounded"
            >
              See pricing
              <ArrowRight className="w-4 h-4" />
            </Link>
          </div>
        </div>
      </section>

      {/* ===== CTA ===== */}
      <section className="py-24 px-6">
        <div className="max-w-2xl mx-auto text-center">
          <h2 className="font-display text-3xl md:text-4xl font-semibold tracking-tight text-ink mb-4">
            Know exactly what to study next
          </h2>
          <p className="text-ink-soft mb-9 text-base">
            Create a course, add your materials, and ask your first question in under two minutes.
          </p>
          <Button size="lg" onClick={toLogin} rightIcon={<ArrowRight className="w-4 h-4" />}>
            Start free
          </Button>
        </div>
      </section>

      <PublicFooter />
    </div>
  )
}
