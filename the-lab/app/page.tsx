import Link from "next/link"
import { Button } from "@/components/ui/button"

const scores = [
  { label: "Skills Match", score: 0.82 },
  { label: "Experience", score: 0.74 },
  { label: "Education", score: 0.91 },
  { label: "Category Fit", score: 0.88 },
  { label: "Semantic Score", score: 0.79 },
]
const overall = 0.83

const tiers = [
  {
    num: "T3",
    name: "Extractor",
    detail: "spaCy NER",
    items: ["Skills", "Keyword Overlap", "Jaccard"],
  },
  {
    num: "T2",
    name: "Category Encoder",
    detail: "DistilBERT",
    items: ["24 Classes", "Confidence", "Penalty"],
  },
  {
    num: "T1",
    name: "Cross-Encoder",
    detail: "BERT-base",
    items: ["Regression", "3-Class Label", "Semantic"],
  },
]

export default function Home() {
  return (
    <>
      {/* ─── Hero ──────────────────────────────────────────── */}
      <section className="relative min-h-dvh flex items-center justify-center overflow-hidden bg-[#0a0a0f]">
        {/* Grid background — slow pan */}
        <div
          className="pointer-events-none absolute inset-0 opacity-[0.05] animate-grid-scroll"
          style={{
            backgroundImage:
              "linear-gradient(rgba(255,255,255,.1) 1px, transparent 1px), linear-gradient(90deg, rgba(255,255,255,.1) 1px, transparent 1px)",
            backgroundSize: "48px 48px",
          }}
        />
        {/* Second grid layer offset for parallax feel */}
        <div
          className="pointer-events-none absolute inset-0 opacity-[0.025]"
          aria-hidden="true"
          style={{
            backgroundImage:
              "linear-gradient(rgba(68,136,255,.08) 1px, transparent 1px), linear-gradient(90deg, rgba(68,136,255,.08) 1px, transparent 1px)",
            backgroundSize: "96px 96px",
          }}
        />

        {/* Glow orbs — floating with larger, more vibrant gradients */}
        <div
          className="pointer-events-none absolute -top-[200px] -right-[120px] h-[600px] w-[600px] animate-float"
          aria-hidden="true"
          style={{
            background: "radial-gradient(circle, rgba(68,136,255,0.3) 0%, rgba(68,136,255,0.12) 40%, transparent 70%)",
          }}
        />
        <div
          className="pointer-events-none absolute -bottom-[150px] -left-[100px] h-[500px] w-[500px] animate-float-slow"
          aria-hidden="true"
          style={{
            background: "radial-gradient(circle, rgba(255,0,51,0.12) 0%, rgba(255,50,80,0.06) 40%, transparent 70%)",
          }}
        />
        <div
          className="pointer-events-none absolute top-[40%] -right-[80px] h-[300px] w-[300px] animate-float-slow"
          aria-hidden="true"
          style={{
            background: "radial-gradient(circle, rgba(0,200,255,0.1) 0%, transparent 60%)",
            animationDelay: "-6s",
          }}
        />


        <div className="relative z-10 w-full max-w-[1100px] px-6 py-12">
          <div className="stagger-delay space-y-8">

            <p className="animate-hero-reveal font-mono text-[13px] text-[#999]">
              TriadRank Research Group &middot; July 2026 &middot; v1.0.0
            </p>

            <h1 className="animate-hero-reveal max-w-[860px] text-[clamp(2.2rem,6vw,4.5rem)] leading-[1.06] font-semibold text-[#F9F9F6] text-balance">
              <span className="text-[#4488FF]">TriadRank:</span> A transparent
              resume scoring pipeline.
            </h1>

            <p className="animate-hero-reveal max-w-[600px] text-[17px] leading-relaxed text-[#ccc]">
              Three transparent ML stages &mdash; entity extraction, category
              validation, deep scoring &mdash; each independently inspectable
              and explainable.
            </p>

            {/* Animated pipeline diagram — styled boxes, no invisible overlays */}
            <div className="animate-hero-reveal group relative rounded border border-[#333] bg-white/[0.02] overflow-hidden transition-all duration-500 hover:border-[#4488FF] hover:bg-white/[0.04] hover:shadow-[0_0_30px_rgba(68,136,255,0.12)]">
              <div className="border-b border-[#333] bg-white/[0.04] px-4 py-2 font-mono text-[11px] text-[#999] flex items-center gap-2">
                <span className="inline-block size-1.5 rounded-full bg-[#4488FF] animate-pulse" />
                pipeline/triadrank.py
              </div>

              <div className="p-5 overflow-x-auto">
                <div className="flex items-stretch gap-0 min-w-[580px]">

                  {/* ── Tier 3 ── */}
                  <div className="flex-1 border border-[#444] bg-white/[0.03] animate-tier-glow-1 transition-all duration-500 hover:border-[#4488FF]/60">
                    <div className="p-3">
                      <div className="text-[10px] font-mono text-[#4488FF] font-semibold tracking-wider mb-1">TIER 3</div>
                      <div className="text-[13px] font-mono text-white font-semibold">Extractor</div>
                      <div className="text-[9px] font-mono text-[#999] mb-2">spaCy NER</div>
                      <div className="h-px bg-[#333] mb-2" />
                      <div className="space-y-0.5">
                        <div className="text-[10px] font-mono text-[#aaa] flex items-center gap-1.5">
                          <span className="inline-block size-1 rounded-full bg-[#4488FF]/70" />Skills
                        </div>
                        <div className="text-[10px] font-mono text-[#aaa] flex items-center gap-1.5">
                          <span className="inline-block size-1 rounded-full bg-[#4488FF]/70" />Keyword Overlap
                        </div>
                        <div className="text-[10px] font-mono text-[#aaa] flex items-center gap-1.5">
                          <span className="inline-block size-1 rounded-full bg-[#4488FF]/70" />Jaccard
                        </div>
                      </div>
                    </div>
                  </div>

                  {/* ── Arrow 1 ── */}
                  <div className="flex items-center justify-center shrink-0 w-7">
                    <svg viewBox="0 0 24 24" className="w-5 h-5 text-[#444]">
                      <path d="M5 12h14M13 5l7 7-7 7" fill="none" stroke="currentColor" strokeWidth="1.5" />
                    </svg>
                  </div>

                  {/* ── Tier 2 ── */}
                  <div className="flex-1 border border-[#444] bg-white/[0.03] animate-tier-glow-2 transition-all duration-500 hover:border-[#4488FF]/60">
                    <div className="p-3">
                      <div className="text-[10px] font-mono text-[#4488FF] font-semibold tracking-wider mb-1">TIER 2</div>
                      <div className="text-[13px] font-mono text-white font-semibold">Category Encoder</div>
                      <div className="text-[9px] font-mono text-[#999] mb-2">DistilBERT</div>
                      <div className="h-px bg-[#333] mb-2" />
                      <div className="space-y-0.5">
                        <div className="text-[10px] font-mono text-[#aaa] flex items-center gap-1.5">
                          <span className="inline-block size-1 rounded-full bg-[#4488FF]/70" />24 Classes
                        </div>
                        <div className="text-[10px] font-mono text-[#aaa] flex items-center gap-1.5">
                          <span className="inline-block size-1 rounded-full bg-[#4488FF]/70" />Confidence Score
                        </div>
                        <div className="text-[10px] font-mono text-[#aaa] flex items-center gap-1.5">
                          <span className="inline-block size-1 rounded-full bg-[#4488FF]/70" />Penalty Logic
                        </div>
                      </div>
                    </div>
                  </div>

                  {/* ── Arrow 2 ── */}
                  <div className="flex items-center justify-center shrink-0 w-7">
                    <svg viewBox="0 0 24 24" className="w-5 h-5 text-[#444]">
                      <path d="M5 12h14M13 5l7 7-7 7" fill="none" stroke="currentColor" strokeWidth="1.5" />
                    </svg>
                  </div>

                  {/* ── Tier 1 ── */}
                  <div className="flex-1 border border-[#444] bg-white/[0.03] animate-tier-glow-3 transition-all duration-500 hover:border-[#4488FF]/60">
                    <div className="p-3">
                      <div className="text-[10px] font-mono text-[#4488FF] font-semibold tracking-wider mb-1">TIER 1</div>
                      <div className="text-[13px] font-mono text-white font-semibold">Cross-Encoder</div>
                      <div className="text-[9px] font-mono text-[#999] mb-2">BERT-base</div>
                      <div className="h-px bg-[#333] mb-2" />
                      <div className="space-y-0.5">
                        <div className="text-[10px] font-mono text-[#aaa] flex items-center gap-1.5">
                          <span className="inline-block size-1 rounded-full bg-[#4488FF]/70" />Regression Score
                        </div>
                        <div className="text-[10px] font-mono text-[#aaa] flex items-center gap-1.5">
                          <span className="inline-block size-1 rounded-full bg-[#4488FF]/70" />3-Class Label
                        </div>
                        <div className="text-[10px] font-mono text-[#aaa] flex items-center gap-1.5">
                          <span className="inline-block size-1 rounded-full bg-[#4488FF]/70" />Semantic Fit
                        </div>
                      </div>
                    </div>
                  </div>

                </div>
              </div>

              <div className="border-t border-[#333] bg-white/[0.02] px-4 py-1.5 font-mono text-[10px] text-[#999] text-right">
                score(candidate, JD) = f<sub>ce</sub> &middot; P(category)
              </div>
            </div>

            <div className="animate-hero-reveal flex flex-wrap gap-4 pt-2">
              <Link href="/sandbox">
                <Button className="h-11 cursor-pointer px-6 text-base shadow-[4px_4px_0px_#111111] transition-all duration-200 hover:-translate-x-0.5 hover:-translate-y-0.5 hover:shadow-[6px_6px_0px_#111] active:translate-x-0 active:translate-y-0">
                  Try the Sandbox
                </Button>
              </Link>
              <Link href="/methodology">
                <Button
                  variant="outline"
                  className="h-11 cursor-pointer border-[#444] bg-transparent px-6 text-base text-[#ccc] shadow-[4px_4px_0px_#000] transition-all duration-200 hover:-translate-x-0.5 hover:-translate-y-0.5 hover:bg-white/10 hover:text-white hover:border-[#666] hover:shadow-[6px_6px_0px_#000] active:translate-x-0 active:translate-y-0"
                >
                  Read the Methodology
                </Button>
              </Link>
            </div>
          </div>
        </div>
      </section>

      {/* ─── Feature sections ─────────────────────────────── */}
      <section className="bg-[#F9F9F6] px-6 py-16">
        <div className="mx-auto max-w-[1100px]">
          <div className="stagger-delay">
            <div className="animate-fade-in-up mb-10">
              <h2 className="text-[clamp(1.5rem,3vw,2.2rem)] font-semibold leading-tight text-balance">
                See exactly how your resume is scored.
              </h2>
              <p className="mt-3 max-w-[580px] text-[15px] leading-relaxed text-muted-foreground">
                Every candidate passes through three scoring tiers. Each tier produces
                interpretable output you can inspect &mdash; no black box.
              </p>
            </div>

            <div className="grid gap-8 lg:grid-cols-3">
              {tiers.map((t, i) => (
                <div
                  key={t.num}
                  className="animate-fade-in-up border border-[#111] bg-white p-6 transition-all duration-300 hover:-translate-y-1 hover:shadow-[6px_6px_0px_#111]"
                  style={{ animationDelay: `${i * 120}ms` }}
                >
                  <div className="mb-4 flex items-center gap-3">
                    <div className="flex h-9 w-9 items-center justify-center bg-[#111] font-mono text-sm font-bold text-white">
                      {t.num.slice(1)}
                    </div>
                    <div>
                      <h3 className="text-base font-semibold">{t.name}</h3>
                      <p className="font-mono text-[11px] text-muted-foreground">
                        {t.detail}
                      </p>
                    </div>
                  </div>
                  <p className="text-sm leading-relaxed text-muted-foreground">
                    {t.num === "T3" && "Parses skills, experience years, education level, and keyword overlaps from raw resume text. Outputs structured entities and a Jaccard similarity score."}
                    {t.num === "T2" && "Classifies the resume into one of 24 job categories. A category mismatch applies a configurable penalty before deep scoring."}
                    {t.num === "T1" && "Computes a semantic fit score (0–1) between resume and job description, producing a three-class label: Good Fit, Potential Fit, or Bad Fit."}
                  </p>
                  <div className="mt-4 flex flex-wrap gap-1.5">
                    {t.items.map((item) => (
                      <span
                        key={item}
                        className="rounded bg-[#f0f0ee] px-2 py-0.5 font-mono text-[10px] text-[#666]"
                      >
                        {item}
                      </span>
                    ))}
                  </div>
                </div>
              ))}
            </div>

            {/* Score preview card */}
            <div className="animate-fade-in-up mt-10 border border-[#111] bg-white p-6 transition-all duration-300 hover:-translate-y-1 hover:shadow-[6px_6px_0px_#111]">
              <div className="mb-4 flex items-center gap-2">
                <span className="inline-block h-2 w-2 rounded-full bg-[#0000FF] animate-pulse" />
                <span className="font-mono text-[11px] uppercase tracking-widest text-muted-foreground">
                  Sample Score Breakdown
                </span>
              </div>
              <div className="grid gap-6 md:grid-cols-2">
                <svg viewBox="0 0 240 240" className="h-56 w-56 justify-self-center md:justify-self-start">
                  <defs>
                    <linearGradient id="radarFill" x1="0%" y1="0%" x2="100%" y2="100%">
                      <stop offset="0%" stopColor="#0000FF" stopOpacity="0.3" />
                      <stop offset="100%" stopColor="#0000FF" stopOpacity="0.05" />
                    </linearGradient>
                  </defs>
                  <polygon points="120,30 205.6,92.2 172.9,192.8 67.1,192.8 34.4,92.2" fill="none" stroke="#ddd" strokeWidth="1" />
                  <polygon points="120,48 188.5,97.8 162.3,178.2 77.7,178.2 51.5,97.8" fill="none" stroke="#ddd" strokeWidth="1" />
                  <polygon points="120,66 171.4,103.3 151.7,163.7 88.3,163.7 68.6,103.3" fill="none" stroke="#ddd" strokeWidth="1" />
                  <polygon points="120,84 154.2,108.9 141.1,149.1 98.9,149.1 85.8,108.9" fill="none" stroke="#ddd" strokeWidth="1" />
                  <polygon points="120,102 137.1,114.4 130.6,134.6 109.4,134.6 102.9,114.4" fill="none" stroke="#ddd" strokeWidth="1" />
                  <line x1="120" y1="120" x2="120" y2="30" stroke="#ddd" strokeWidth="1" />
                  <line x1="120" y1="120" x2="205.6" y2="92.2" stroke="#ddd" strokeWidth="1" />
                  <line x1="120" y1="120" x2="172.9" y2="192.8" stroke="#ddd" strokeWidth="1" />
                  <line x1="120" y1="120" x2="67.1" y2="192.8" stroke="#ddd" strokeWidth="1" />
                  <line x1="120" y1="120" x2="34.4" y2="92.2" stroke="#ddd" strokeWidth="1" />
                  <polygon points="120,46.2 183.3,99.4 168.1,186.3 73.4,184.1 52.4,98.0" fill="url(#radarFill)" stroke="#0000FF" strokeWidth="2" />
                  {[{x:120,y:46.2},{x:183.3,y:99.4},{x:168.1,y:186.3},{x:73.4,y:184.1},{x:52.4,y:98.0}].map((p, i) => (
                    <circle key={i} cx={p.x} cy={p.y} r="3" fill="#0000FF" />
                  ))}
                  <text x="120" y="22" textAnchor="middle" fill="#999" fontSize="8" fontFamily="monospace">Skills</text>
                  <text x="214" y="92" textAnchor="start" fill="#999" fontSize="8" fontFamily="monospace">Experience</text>
                  <text x="178" y="205" textAnchor="middle" fill="#999" fontSize="8" fontFamily="monospace">Education</text>
                  <text x="62" y="205" textAnchor="middle" fill="#999" fontSize="8" fontFamily="monospace">Category</text>
                  <text x="28" y="92" textAnchor="end" fill="#999" fontSize="8" fontFamily="monospace">Semantic</text>
                </svg>

                <table className="w-full self-center font-mono text-[13px]">
                  <tbody>
                    {scores.map((s) => (
                      <tr key={s.label}>
                        <td className="whitespace-nowrap py-2 pr-4 text-muted-foreground">{s.label}</td>
                        <td className="w-full py-2 pr-4">
                          <div className="h-2 w-full bg-[#f0f0ee]">
                            <div className="animate-score-bar h-full bg-[#0000FF]" style={{ width: `${Math.round(s.score * 100)}%`, animationDelay: '0.5s' }} />
                          </div>
                        </td>
                        <td className="py-2 text-right tabular-nums font-semibold text-[#111]">{s.score.toFixed(2)}</td>
                      </tr>
                    ))}
                  </tbody>
                  <tfoot>
                    <tr className="border-t border-[#111]">
                      <td className="py-3 pr-4 font-semibold text-[#111]">Overall</td>
                      <td className="py-3 pr-4">
                        <div className="h-2 w-full bg-[#f0f0ee]">
                          <div className="animate-score-bar h-full bg-[#0000FF]" style={{ width: `${Math.round(overall * 100)}%`, animationDelay: '0.7s' }} />
                        </div>
                      </td>
                      <td className="py-3 text-right tabular-nums font-bold text-[#0000FF]">{overall.toFixed(2)}</td>
                    </tr>
                  </tfoot>
                </table>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* ─── Footer ───────────────────────────────────────── */}
      <footer className="relative border-t border-[#111] bg-[#F9F9F6] overflow-hidden">
        <div
          className="pointer-events-none absolute inset-0 opacity-[0.02]"
          style={{
            backgroundImage:
              "linear-gradient(rgba(0,0,0,.1) 1px, transparent 1px), linear-gradient(90deg, rgba(0,0,0,.1) 1px, transparent 1px)",
            backgroundSize: "32px 32px",
          }}
        />
        <div className="relative mx-auto max-w-[1100px] px-6 py-8">
          <div className="flex flex-col gap-6 sm:flex-row sm:items-center sm:justify-between">
            {/* Left */}
            <div>
              <div className="flex items-center gap-2">
                <span className="inline-block size-2 rounded-full bg-[#0000FF]" />
                <span className="font-mono text-[13px] font-semibold text-[#111]">The Lab</span>
              </div>
              <p className="mt-1 font-mono text-[11px] text-muted-foreground">
                TriadRank ATS &middot; v1.0.0
              </p>
            </div>

            {/* Center */}
            <div className="flex flex-wrap gap-x-6 gap-y-1">
              <Link href="/sandbox" className="font-mono text-[12px] text-muted-foreground transition-colors hover:text-[#0000FF]">
                Sandbox
              </Link>
              <Link href="/results" className="font-mono text-[12px] text-muted-foreground transition-colors hover:text-[#0000FF]">
                Results
              </Link>
              <Link href="/methodology" className="font-mono text-[12px] text-muted-foreground transition-colors hover:text-[#0000FF]">
                Methodology
              </Link>
            </div>

            {/* Right */}
            <div className="flex items-center gap-3">
              <span className="inline-flex items-center gap-1.5 rounded border border-[#ddd] bg-white px-2 py-1 font-mono text-[10px] text-muted-foreground">
                <span className="inline-block size-1.5 rounded-full bg-[#00AA00] shadow-[0_0_4px_rgba(0,170,0,0.4)]" />
                3-tier pipeline
              </span>
              <span className="font-mono text-[10px] text-muted-foreground">
                24 categories
              </span>
            </div>
          </div>

          <div className="mt-6 border-t border-[#eee] pt-4">
            <p className="text-center font-mono text-[10px] text-muted-foreground/60">
              Built with Next.js &middot; PyTorch &middot; Transformers &middot; spaCy
            </p>
          </div>
        </div>
      </footer>
    </>
  )
}
