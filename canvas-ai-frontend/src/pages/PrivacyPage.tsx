import { LegalShell, LegalSection } from '@/components/marketing/LegalDoc'

export default function PrivacyPage() {
  return (
    <LegalShell title="Privacy Policy" updated="July 2026 (draft)">
      <LegalSection num="01" title="Overview">
        <p>
          This policy explains what Vindexa collects, how it is used, and the choices you have. The
          short version: <strong>your course materials are used only to power your own study tools.</strong> We
          do not sell your data, show you ads, or share your content with other users.
        </p>
      </LegalSection>

      <LegalSection num="02" title="What we collect">
        <ul className="list-disc pl-5 space-y-1.5">
          <li><strong>Account information</strong> — your email address and authentication credentials.</li>
          <li><strong>Course materials</strong> — files you upload (PDF, DOCX, PPTX) or import from Canvas LMS, including syllabi and exam dates.</li>
          <li><strong>Study activity</strong> — your questions, quiz and practice answers, confidence ratings, exam attempts, notes and review history. This is what makes readiness scores and adaptive practice work.</li>
          <li><strong>Technical data</strong> — basic logs (timestamps, request metadata) needed to operate and secure the Service.</li>
        </ul>
      </LegalSection>

      <LegalSection num="03" title="How your data is used">
        <p>
          Everything we collect serves one purpose: providing the Service to you. Your materials are
          indexed so the Service can answer your questions with citations; your study activity drives
          your practice, review queue and readiness scores; your email is used for authentication and
          essential service messages.
        </p>
      </LegalSection>

      <LegalSection num="04" title="Where your data lives">
        <p>
          Your files, study data and account information are stored in <strong>Supabase</strong> (managed
          Postgres and file storage). Data is isolated per user: your content is scoped to your account
          and is not visible to, or blended with, any other user&rsquo;s data.
        </p>
      </LegalSection>

      <LegalSection num="05" title="AI processing">
        <p>
          To generate answers, quizzes, notes and other study tools, relevant excerpts of your
          materials and your prompts are sent to the <strong>Anthropic API</strong> for processing. This
          processing happens only to produce your requested output. We do not use your content to
          train models of our own, and we do not permit it to be used to train third-party models.
        </p>
      </LegalSection>

      <LegalSection num="06" title="What we do not do">
        <ul className="list-disc pl-5 space-y-1.5">
          <li>We do not sell or rent your personal data or your course materials.</li>
          <li>We do not serve advertising or share data with ad networks.</li>
          <li>We do not share your content with other users or use it to build features for them.</li>
        </ul>
      </LegalSection>

      <LegalSection num="07" title="Retention and deletion">
        <p>
          Your content is retained while your account is active so your study history keeps working
          for you. You can delete individual files or courses at any time, and deleting your account
          removes your content from active systems within a reasonable period, except where limited
          retention is required by law or for security.
        </p>
      </LegalSection>

      <LegalSection num="08" title="Security">
        <p>
          Access to your data is protected by authentication and per-user isolation at the database
          level. Data is encrypted in transit. No system is perfectly secure, but we design so that a
          user can only ever reach their own data.
        </p>
      </LegalSection>

      <LegalSection num="09" title="Your rights">
        <p>
          Depending on where you live, you may have rights to access, correct, export or delete your
          personal data. You can exercise most of these directly in the app (deleting files, courses
          or your account), or by contacting us.
        </p>
      </LegalSection>

      <LegalSection num="10" title="Changes to this policy">
        <p>
          If this policy changes in a material way, we will notify you by email or an in-app notice
          before the change takes effect.
        </p>
      </LegalSection>

      <LegalSection num="11" title="Contact">
        <p>
          Privacy questions: <a href="mailto:hello@vindexa.app" className="text-accent hover:text-accent-deep">hello@vindexa.app</a>.
        </p>
      </LegalSection>
    </LegalShell>
  )
}
