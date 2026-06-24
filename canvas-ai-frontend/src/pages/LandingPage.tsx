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
} from 'lucide-react'

const features = [
  {
    icon: MessageCircle,
    title: 'Grounded AI Chat',
    description: 'Ask anything about your course and get precise answers cited to your actual materials — not the open internet.',
  },
  {
    icon: GraduationCap,
    title: 'Socratic Tutor',
    description: 'A tutor that guides you to the answer with questions instead of handing it over — so the understanding sticks.',
  },
  {
    icon: Target,
    title: 'Adaptive Practice & Quizzes',
    description: 'Problem sets and quizzes that adjust to your level and zero in on the topics you keep getting wrong.',
  },
  {
    icon: ClipboardList,
    title: 'Exam Simulator',
    description: 'Timed mock exams generated from your materials, plus a readiness score that tells you when you’re ready.',
  },
  {
    icon: RefreshCw,
    title: 'Smart Review',
    description: 'Every mistake feeds a spaced-repetition queue that resurfaces exactly what you’re weakest on, right before you forget it.',
  },
  {
    icon: FileText,
    title: 'Notes, Flashcards & Audio',
    description: 'Turn lectures into study notes, auto-generated flashcards, and audio overviews you can review anywhere.',
  },
]

const steps = [
  {
    icon: Upload,
    number: '01',
    title: 'Upload your materials',
    description: 'Drop in your PDFs, lecture slides, and documents. We support all common course file formats.',
  },
  {
    icon: Brain,
    number: '02',
    title: 'AI processes everything',
    description: 'Our AI reads and understands your entire course content, building a deep knowledge base.',
  },
  {
    icon: Zap,
    number: '03',
    title: 'Study with superpowers',
    description: 'Access quizzes, practice problems, notes, exams, and more — all tailored to your exact course.',
  },
]

const stats = [
  { value: '10+', label: 'Study tools' },
  { value: '< 30s', label: 'To generate a quiz' },
  { value: 'Any', label: 'Course or subject' },
  { value: 'Free', label: 'To get started' },
]

const benefits = [
  'Generates content from your actual course materials',
  'Adaptive difficulty that matches your level',
  'Instant feedback with detailed explanations',
  'Works with PDFs, DOCX, and PPTX files',
  'Track progress and identify weak areas',
  'Study on any device, anytime',
]

export default function LandingPage() {
  const navigate = useNavigate()

  return (
    <div className="relative w-full text-white">
      {/* All scrollable content */}
      <div className="relative z-10">
        {/* Nav */}
        <nav className="fixed top-0 left-0 right-0 z-50 bg-black/40 backdrop-blur-md border-b border-white/[0.06]">
          <div className="flex items-center justify-between px-6 py-3 max-w-6xl mx-auto">
            <div className="flex items-center gap-2.5">
              <img src="/favicon-32x32.png" alt="Vindexa" className="w-7 h-7 rounded" />
              <span className="text-base font-semibold tracking-tight">Vindexa</span>
            </div>
            <button
              onClick={() => navigate('/dashboard')}
              className="bg-white/10 hover:bg-white/15 backdrop-blur-sm border border-white/10 text-white px-4 py-1.5 rounded-lg text-sm font-medium transition-colors"
            >
              Open App
            </button>
          </div>
        </nav>

        {/* ===== HERO ===== */}
        <section className="min-h-screen flex flex-col items-center justify-center px-6 pt-16">
          {/* Trust badge */}
          <div className="mb-8 animate-hero-fade-down">
            <div className="flex items-center gap-2 px-6 py-3 bg-cyan-500/10 backdrop-blur-md border border-cyan-300/30 rounded-full text-sm">
              <span>✦</span>
              <span className="text-cyan-100">AI-Powered Study Companion</span>
            </div>
          </div>

          <div className="text-center space-y-6 max-w-5xl mx-auto px-4">
            {/* Headline */}
            <div className="space-y-2">
              <h1 className="text-5xl md:text-7xl lg:text-8xl font-bold bg-gradient-to-r from-blue-300 via-cyan-400 to-teal-300 bg-clip-text text-transparent animate-hero-fade-up hero-delay-200">
                Study Smarter
              </h1>
              <h1 className="text-5xl md:text-7xl lg:text-8xl font-bold bg-gradient-to-r from-cyan-300 via-blue-400 to-indigo-400 bg-clip-text text-transparent animate-hero-fade-up hero-delay-400">
                With Vindexa
              </h1>
            </div>

            {/* Subtitle */}
            <div className="max-w-3xl mx-auto animate-hero-fade-up hero-delay-600">
              <p className="text-lg md:text-xl lg:text-2xl text-cyan-100/90 font-light leading-relaxed">
                Upload your course materials and get AI-generated quizzes, practice problems, notes, and exams — instantly.
              </p>
            </div>

            {/* Buttons */}
            <div className="flex flex-col sm:flex-row gap-4 justify-center mt-10 animate-hero-fade-up hero-delay-800">
              <button
                onClick={() => navigate('/dashboard')}
                className="px-8 py-4 bg-gradient-to-r from-blue-500 to-cyan-500 hover:from-blue-600 hover:to-cyan-600 text-white rounded-full font-semibold text-lg transition-all duration-300 hover:scale-105 hover:shadow-xl hover:shadow-cyan-500/25"
              >
                Get Started Free
              </button>
              <button
                onClick={() => document.getElementById('stats-section')?.scrollIntoView({ behavior: 'smooth' })}
                className="px-8 py-4 bg-cyan-500/10 hover:bg-cyan-500/20 border border-cyan-300/30 hover:border-cyan-300/50 text-cyan-100 rounded-full font-semibold text-lg transition-all duration-300 hover:scale-105 backdrop-blur-sm"
              >
                Learn More
              </button>
            </div>
          </div>

          {/* Scroll hint */}
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 0.4 }}
            transition={{ delay: 1.5 }}
            className="mt-16"
          >
            <div className="w-5 h-8 rounded-full border border-white/20 flex items-start justify-center p-1.5">
              <motion.div
                animate={{ y: [0, 8, 0] }}
                transition={{ duration: 1.5, repeat: Infinity, ease: 'easeInOut' }}
                className="w-1 h-1.5 rounded-full bg-white/60"
              />
            </div>
          </motion.div>
        </section>

        {/* Content sections with heavy overlay so shader peeks through subtly */}
        <div className="relative bg-black/60">

        {/* ===== STATS BAR ===== */}
        <section id="stats-section" className="relative py-12 px-6">
          <div className="max-w-4xl mx-auto">
            <motion.div
              initial={{ opacity: 0, y: 16 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true, margin: '-80px' }}
              transition={{ duration: 0.5 }}
              className="grid grid-cols-2 md:grid-cols-4 gap-6 py-8 px-8 rounded-2xl bg-zinc-900/90 border border-zinc-700/50"
            >
              {stats.map((stat) => (
                <div key={stat.label} className="text-center">
                  <div className="text-2xl md:text-3xl font-bold text-white">{stat.value}</div>
                  <div className="text-xs text-zinc-500 mt-1">{stat.label}</div>
                </div>
              ))}
            </motion.div>
          </div>
        </section>

        {/* ===== HOW IT WORKS ===== */}
        <section className="relative py-24 px-6">
          <div className="max-w-5xl mx-auto">
            <motion.div
              initial={{ opacity: 0, y: 12 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true, margin: '-80px' }}
              transition={{ duration: 0.5 }}
              className="text-center mb-16"
            >
              <p className="text-xs font-medium uppercase tracking-widest text-zinc-500 mb-3">How it works</p>
              <h2 className="text-3xl md:text-4xl font-bold tracking-tight">
                Three steps to better grades
              </h2>
            </motion.div>

            <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
              {steps.map((step, i) => (
                <motion.div
                  key={step.number}
                  initial={{ opacity: 0, y: 20 }}
                  whileInView={{ opacity: 1, y: 0 }}
                  viewport={{ once: true, margin: '-60px' }}
                  transition={{ duration: 0.45, delay: i * 0.1 }}
                  className="relative p-6 rounded-2xl bg-zinc-900/90 border border-zinc-700/50"
                >
                  <span className="text-[4rem] font-black text-white/[0.04] absolute top-3 right-5 leading-none select-none">
                    {step.number}
                  </span>
                  <div className="w-10 h-10 rounded-xl bg-white/[0.06] border border-white/[0.08] flex items-center justify-center mb-5">
                    <step.icon className="w-5 h-5 text-white/60" />
                  </div>
                  <h3 className="text-base font-semibold text-white mb-2">{step.title}</h3>
                  <p className="text-sm text-zinc-400 leading-relaxed">{step.description}</p>
                </motion.div>
              ))}
            </div>
          </div>
        </section>

        {/* ===== FEATURES ===== */}
        <section className="relative py-24 px-6">
          <div className="max-w-5xl mx-auto">
            <motion.div
              initial={{ opacity: 0, y: 12 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true, margin: '-80px' }}
              transition={{ duration: 0.5 }}
              className="text-center mb-16"
            >
              <p className="text-xs font-medium uppercase tracking-widest text-zinc-500 mb-3">Features</p>
              <h2 className="text-3xl md:text-4xl font-bold tracking-tight">
                Everything you need to excel
              </h2>
              <p className="mt-4 text-zinc-400 text-base max-w-lg mx-auto">
                A connected study system where every tool feeds the next.
              </p>
            </motion.div>

            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
              {features.map((feature, i) => (
                <motion.div
                  key={feature.title}
                  initial={{ opacity: 0, y: 20 }}
                  whileInView={{ opacity: 1, y: 0 }}
                  viewport={{ once: true, margin: '-40px' }}
                  transition={{ duration: 0.4, delay: i * 0.06 }}
                  className="group p-6 rounded-2xl bg-zinc-900/90 border border-zinc-700/50 hover:bg-zinc-800/90 hover:border-zinc-600/60 transition-all"
                >
                  <feature.icon className="w-5 h-5 text-zinc-500 group-hover:text-cyan-400 transition-colors mb-4" />
                  <h3 className="text-sm font-semibold text-white mb-2">{feature.title}</h3>
                  <p className="text-sm text-zinc-400 leading-relaxed">{feature.description}</p>
                </motion.div>
              ))}
            </div>
          </div>
        </section>

        {/* ===== WHY VINDEXA ===== */}
        <section className="relative py-24 px-6">
          <div className="max-w-4xl mx-auto">
            <div className="grid grid-cols-1 md:grid-cols-2 gap-12 items-center">
              <motion.div
                initial={{ opacity: 0, x: -20 }}
                whileInView={{ opacity: 1, x: 0 }}
                viewport={{ once: true, margin: '-80px' }}
                transition={{ duration: 0.5 }}
              >
                <p className="text-xs font-medium uppercase tracking-widest text-zinc-500 mb-3">Why Vindexa</p>
                <h2 className="text-3xl md:text-4xl font-bold tracking-tight mb-4">
                  Built for how
                  <br />
                  students actually study
                </h2>
                <p className="text-zinc-400 text-sm leading-relaxed">
                  Unlike generic AI tools, Vindexa understands your specific course content — and connects it all. Every wrong answer feeds your review queue, your weak topics reshape your practice, and your exam-readiness updates as you go. You always know exactly what to study next.
                </p>
              </motion.div>

              <motion.div
                initial={{ opacity: 0, x: 20 }}
                whileInView={{ opacity: 1, x: 0 }}
                viewport={{ once: true, margin: '-80px' }}
                transition={{ duration: 0.5, delay: 0.1 }}
                className="space-y-3"
              >
                {benefits.map((benefit, i) => (
                  <motion.div
                    key={i}
                    initial={{ opacity: 0, x: 12 }}
                    whileInView={{ opacity: 1, x: 0 }}
                    viewport={{ once: true }}
                    transition={{ duration: 0.3, delay: 0.15 + i * 0.05 }}
                    className="flex items-start gap-3 p-3 rounded-xl bg-zinc-900/90 border border-zinc-700/50"
                  >
                    <CheckCircle className="w-4 h-4 text-cyan-400/70 mt-0.5 flex-shrink-0" />
                    <span className="text-sm text-zinc-400">{benefit}</span>
                  </motion.div>
                ))}
              </motion.div>
            </div>
          </div>
        </section>

        {/* ===== BOTTOM CTA ===== */}
        <section className="relative py-32 px-6">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true, margin: '-60px' }}
            transition={{ duration: 0.5 }}
            className="max-w-2xl mx-auto text-center"
          >
            <h2 className="text-3xl md:text-5xl font-bold tracking-tight mb-4">
              Start studying smarter
            </h2>
            <p className="text-zinc-400 mb-10 text-base">
              Upload your first course and see the difference in minutes. No credit card required.
            </p>
            <div className="flex flex-col sm:flex-row items-center justify-center gap-4">
              <button
                onClick={() => navigate('/dashboard')}
                className="px-8 py-4 bg-gradient-to-r from-blue-500 to-cyan-500 hover:from-blue-600 hover:to-cyan-600 text-white rounded-full font-semibold text-lg transition-all duration-300 hover:scale-105 hover:shadow-xl hover:shadow-cyan-500/25"
              >
                Get Started Free
              </button>
            </div>
          </motion.div>
        </section>

        {/* ===== FOOTER ===== */}
        </div>
        <footer className="relative border-t border-white/[0.06] py-8 bg-black/80 backdrop-blur-xl">
          <div className="max-w-6xl mx-auto px-6 flex flex-col sm:flex-row items-center justify-between gap-4">
            <div className="flex items-center gap-2">
              <img src="/favicon-32x32.png" alt="Vindexa" className="w-5 h-5 rounded opacity-40" />
              <span className="text-xs text-white/25">Vindexa</span>
            </div>
            <div className="flex items-center gap-6">
              <button
                onClick={() => navigate('/dashboard')}
                className="text-xs text-white/25 hover:text-white/50 transition-colors"
              >
                Dashboard
              </button>
              <button
                onClick={() => navigate('/settings')}
                className="text-xs text-white/25 hover:text-white/50 transition-colors"
              >
                Settings
              </button>
            </div>
            <span className="text-xs text-white/15">Built for students, by students.</span>
          </div>
        </footer>
      </div>
    </div>
  )
}
