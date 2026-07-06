import { useNavigate } from 'react-router-dom'
import { motion } from 'motion/react'
import {
  MessageCircle,
  GraduationCap,
  Target,
  ClipboardList,
  RefreshCw,
  FileText,
  Upload,
  Brain,
  Zap,
  CheckCircle,
  ArrowRight,
  BookOpen,
} from 'lucide-react'
import { BrandMark } from '@/components/ui/BrandMark'
import { Button } from '@/components/ui/Button'
import { Badge } from '@/components/ui/Badge'
import { ProgressRing } from '@/components/ui/Progress'

const FEATURES = [
  {
    icon: MessageCircle,
    title: 'Answers with page citations',
    description: 'Ask anything about your course. Every answer is retrieved from your own files and cites the exact source — no internet guesswork.',
  },
  {
    icon: GraduationCap,
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
    icon: Brain,
    number: '02',
    title: 'Vindexa indexes it',
    description: 'Your materials are chunked, embedded and cross-linked into a course-specific knowledge base.',
  },
  {
    icon: Zap,
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

const sectionReveal = {
  initial: { opacity: 0, y: 12 },
  whileInView: { opacity: 1, y: 0 },
  viewport: { once: true, margin: '-80px' },
  transition: { duration: 0.35, ease: [0.22, 1, 0.36, 1] as const },
}

/** DOM-built product preview — always crisp, always current with the brand. */
function ProductPreview() {
  return (
    <div className="relative mx-auto max-w-4xl rounded-2xl border border-border bg-bg-card elev-3 overflow-hidden text-left">
      {/* Window chrome */}
      <div className="flex items-center gap-1.5 px-4 h-9 border-b border-border-subtle bg-bg-subtle">
        <span className="w-2.5 h-2.5 rounded-full bg-white/10" />
        <span className="w-2.5 h-2.5 rounded-full bg-white/10" />
        <span className="w-2.5 h-2.5 rounded-full bg-white/10" />
        <span className="ml-3 text-[11px] text-zinc-500 truncate">Data Structures · Learn</span>
      </div>
      <div className="grid grid-cols-1 md:grid-cols-[1fr_220px]">
        {/* Chat pane */}
        <div className="p-5 sm:p-6 space-y-4 min-w-0">
          <div className="flex justify-end">
            <div className="rounded-xl rounded-br-sm bg-white/[0.06] border border-white/10 px-3.5 py-2 text-sm text-zinc-200 max-w-[85%]">
              When should I use Dijkstra instead of A*?
            </div>
          </div>
          <div className="max-w-[92%]">
            <p className="text-sm text-zinc-300 leading-relaxed">
              Use Dijkstra when you have no admissible heuristic or need shortest paths to
              <span className="text-zinc-100 font-medium"> every</span> node; A* wins when a good heuristic
              (like straight-line distance) can steer the search toward a single goal…
            </p>
            <div className="flex flex-wrap items-center gap-2 mt-3">
              <Badge tone="accent" icon={<BookOpen />}>Lecture 8 · pp. 12–14</Badge>
              <Badge tone="neutral">Graphs.pdf · p. 3</Badge>
            </div>
          </div>
          <div className="flex flex-wrap gap-2 pt-1">
            {['Quiz me on this', 'Show a worked example'].map((chip) => (
              <span key={chip} className="text-xs text-zinc-400 border border-border rounded-full px-3 py-1">
                {chip}
              </span>
            ))}
          </div>
        </div>
        {/* Readiness rail */}
        <div className="border-t md:border-t-0 md:border-l border-border-subtle p-5 bg-bg-subtle/60">
          <p className="text-[10px] font-semibold uppercase tracking-[0.18em] text-zinc-500 mb-3">Exam readiness</p>
          <div className="flex md:flex-col items-center md:items-start gap-4">
            <ProgressRing value={72} size={84} strokeWidth={7}>
              <span className="text-lg font-bold text-emerald-300">72%</span>
            </ProgressRing>
            <div className="space-y-1.5">
              <p className="text-[10px] font-semibold uppercase tracking-[0.18em] text-zinc-500">Focus next</p>
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
    <div className="relative w-full text-zinc-50">
      {/* Nav */}
      <nav className="fixed top-0 left-0 right-0 z-50 glass-bar">
        <div className="flex items-center justify-between px-6 h-14 max-w-6xl mx-auto">
          <div className="flex items-center gap-2.5">
            <BrandMark className="w-7 h-7" />
            <span className="text-base font-semibold tracking-tight">Vindexa</span>
          </div>
          <div className="flex items-center gap-2">
            <Button variant="ghost" size="sm" onClick={toLogin}>Sign in</Button>
            <Button size="sm" onClick={toLogin}>Get started</Button>
          </div>
        </div>
      </nav>

      {/* ===== HERO ===== */}
      <section className="px-6 pt-36 pb-20 sm:pt-44">
        <div className="max-w-3xl mx-auto text-center">
          <h1 className="text-4xl sm:text-5xl md:text-6xl font-semibold tracking-tight leading-[1.05] animate-hero-fade-up">
            Your course materials,
            <br />
            turned into a <span className="text-gradient-brand">study system</span>
          </h1>
          <p className="mt-6 text-lg text-zinc-400 leading-relaxed max-w-2xl mx-auto animate-hero-fade-up hero-delay-200">
            Vindexa indexes your slides, readings and syllabus — then answers with page-level
            citations, drills your weak topics, and tells you when you’re ready for the exam.
          </p>
          <div className="flex flex-col sm:flex-row gap-3 justify-center mt-9 animate-hero-fade-up hero-delay-400">
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
        <div className="mt-16 px-0 sm:px-6 animate-hero-fade-up hero-delay-600">
          <ProductPreview />
        </div>
      </section>

      {/* ===== HOW IT WORKS ===== */}
      <section id="how-it-works" className="py-24 px-6">
        <div className="max-w-5xl mx-auto">
          <motion.div {...sectionReveal} className="text-center mb-14">
            <p className="text-[11px] font-semibold uppercase tracking-[0.2em] text-gradient-brand mb-3">How it works</p>
            <h2 className="text-3xl md:text-4xl font-semibold tracking-tight">From file dump to study plan</h2>
          </motion.div>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            {STEPS.map((step, i) => (
              <motion.div
                key={step.number}
                {...sectionReveal}
                transition={{ ...sectionReveal.transition, delay: i * 0.07 }}
                className="relative p-6 rounded-xl card-surface accent-top"
              >
                <span className="text-[3.5rem] font-black text-white/[0.04] absolute top-3 right-5 leading-none select-none">
                  {step.number}
                </span>
                <div className="w-10 h-10 rounded-xl bg-white/[0.05] border border-white/10 flex items-center justify-center mb-5">
                  <step.icon className="w-5 h-5 text-cyan-300" />
                </div>
                <h3 className="text-base font-semibold text-zinc-50 mb-2">{step.title}</h3>
                <p className="text-sm text-zinc-400 leading-relaxed">{step.description}</p>
              </motion.div>
            ))}
          </div>
        </div>
      </section>

      {/* ===== FEATURES ===== */}
      <section className="py-24 px-6">
        <div className="max-w-5xl mx-auto">
          <motion.div {...sectionReveal} className="text-center mb-14">
            <p className="text-[11px] font-semibold uppercase tracking-[0.2em] text-gradient-brand mb-3">Features</p>
            <h2 className="text-3xl md:text-4xl font-semibold tracking-tight">One connected system</h2>
            <p className="mt-4 text-zinc-400 text-base max-w-lg mx-auto">
              Every tool feeds the next: chat surfaces gaps, practice targets them, review locks them in.
            </p>
          </motion.div>
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
            {FEATURES.map((feature, i) => (
              <motion.div
                key={feature.title}
                {...sectionReveal}
                transition={{ ...sectionReveal.transition, delay: (i % 3) * 0.06 }}
                className="group p-6 rounded-xl card-surface card-interactive"
              >
                <feature.icon className="w-5 h-5 text-zinc-500 group-hover:text-cyan-300 transition-colors mb-4" />
                <h3 className="text-sm font-semibold text-zinc-50 mb-2">{feature.title}</h3>
                <p className="text-sm text-zinc-400 leading-relaxed">{feature.description}</p>
              </motion.div>
            ))}
          </div>
        </div>
      </section>

      {/* ===== WHY ===== */}
      <section className="py-24 px-6">
        <div className="max-w-4xl mx-auto">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-12 items-center">
            <motion.div {...sectionReveal}>
              <p className="text-[11px] font-semibold uppercase tracking-[0.2em] text-gradient-brand mb-3">Why Vindexa</p>
              <h2 className="text-3xl md:text-4xl font-semibold tracking-tight mb-4">
                Built for how students actually study
              </h2>
              <p className="text-zinc-400 text-sm leading-relaxed">
                Generic AI tools answer from the open internet and forget you between sessions.
                Vindexa is grounded in your specific course and remembers every answer you get
                wrong — so the system always knows your weakest topic, and so do you.
              </p>
            </motion.div>
            <motion.div {...sectionReveal} transition={{ ...sectionReveal.transition, delay: 0.08 }} className="space-y-3">
              {BENEFITS.map((benefit) => (
                <div key={benefit} className="flex items-start gap-3 p-3.5 rounded-xl card-surface">
                  <CheckCircle className="w-4 h-4 text-emerald-300 mt-0.5 flex-shrink-0" />
                  <span className="text-sm text-zinc-300">{benefit}</span>
                </div>
              ))}
            </motion.div>
          </div>
        </div>
      </section>

      {/* ===== CTA ===== */}
      <section className="py-28 px-6">
        <motion.div {...sectionReveal} className="max-w-2xl mx-auto text-center">
          <h2 className="text-3xl md:text-4xl font-semibold tracking-tight mb-4">
            Know exactly what to study next
          </h2>
          <p className="text-zinc-400 mb-9 text-base">
            Create a course, add your materials, and ask your first question in under two minutes.
          </p>
          <Button size="lg" onClick={toLogin} rightIcon={<ArrowRight className="w-4 h-4" />}>
            Start free
          </Button>
        </motion.div>
      </section>

      {/* ===== FOOTER ===== */}
      <footer className="border-t border-border-subtle py-8">
        <div className="max-w-6xl mx-auto px-6 flex flex-col sm:flex-row items-center justify-between gap-4">
          <div className="flex items-center gap-2">
            <BrandMark className="w-5 h-5 opacity-50" />
            <span className="text-xs text-zinc-500">Vindexa</span>
          </div>
          <span className="text-xs text-zinc-600">Grounded in your materials. Nothing else.</span>
        </div>
      </footer>
    </div>
  )
}
