import { useParams } from 'react-router-dom'
import { Headphones } from 'lucide-react'
import { Card, PageHeader } from '@/components/ui/Card'

export default function AudioPage() {
  const { courseId: _courseId } = useParams<{ courseId: string }>()

  return (
    <div className="max-w-4xl mx-auto px-6 py-8 space-y-6">
      <PageHeader
        eyebrow="Audio"
        title="Audio Overview"
        subtitle="Listen to AI-generated summaries of your course materials"
      />

      <Card accent padding="none" className="py-12 px-8 text-center">
        <div className="w-14 h-14 rounded-2xl bg-gradient-brand-soft border border-cyan-500/20 flex items-center justify-center mx-auto mb-4">
          <Headphones className="w-7 h-7 text-cyan-300" />
        </div>
        <div className="inline-flex items-center gap-1.5 px-2.5 py-1 bg-gradient-brand-soft border border-cyan-500/20 text-cyan-300 text-xs font-medium rounded-full mb-3">
          Coming soon
        </div>
        <h3 className="text-lg font-semibold text-zinc-100 mb-2">Audio summaries are on the way</h3>
        <p className="text-sm text-zinc-500 max-w-md mx-auto">
          We're building the ability to generate audio overviews of your course materials so you can study on the go. Stay tuned.
        </p>
      </Card>
    </div>
  )
}
