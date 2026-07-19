import Link from 'next/link'
import CodeBlock from '@/components/code-block'
import MobileToc from '@/components/mobile-toc'

const tocLinks = [
  { href: '#overview', label: '1. Overview' },
  { href: '#architecture', label: '2. Architecture' },
  { href: '#tier3', label: '3. Tier 3: Extractor' },
  { href: '#tier2', label: '4. Tier 2: Category Encoder' },
  { href: '#tier1', label: '5. Tier 1: Cross-Encoder' },
  { href: '#scoring', label: '6. Scoring & Ranking' },
  { href: '#bias', label: '7. Bias Mitigation' },
  { href: '#references', label: '8. References' },
]

export default function MethodologyPage() {

  return (
    <div className="mx-auto flex justify-center">
      {/* ─── Sidebar TOC ──────────────────────────────────────── */}
      <aside className="sticky top-[56px] hidden h-[calc(100vh-56px)] w-[250px] shrink-0 overflow-y-auto border-r border-[#111] p-6 lg:block">
        <nav>
          <h2 className="mb-4 font-mono text-xs uppercase tracking-[0.15em]" style={{ color: '#666' }}>
            Contents
          </h2>
          <ul className="space-y-3">
            {tocLinks.map((link) => (
              <li key={link.href}>
                <Link href={link.href} className="font-mono text-[13px] text-[#111] hover:text-blue-600 hover:underline">
                  {link.label}
                </Link>
              </li>
            ))}
          </ul>
        </nav>

        {/* Metadata block */}
        <div className="mt-8 border-t border-[#111] pt-4">
          <dl className="space-y-2">
            <div className="flex justify-between">
              <dt className="font-mono text-[11px] uppercase tracking-[0.1em]" style={{ color: '#666' }}>
                Version
              </dt>
              <dd className="font-mono text-[11px] text-[#111]">v1.0.0</dd>
            </div>
            <div className="flex justify-between">
              <dt className="font-mono text-[11px] uppercase tracking-[0.1em]" style={{ color: '#666' }}>
                Status
              </dt>
              <dd className="font-mono text-[11px] text-[#111]">Peer Review</dd>
            </div>
            <div className="flex justify-between">
              <dt className="font-mono text-[11px] uppercase tracking-[0.1em]" style={{ color: '#666' }}>
                Date
              </dt>
              <dd className="font-mono text-[11px] text-[#111]">2026-07</dd>
            </div>
          </dl>
        </div>
      </aside>

      {/* ─── Main Content ───────────────────────────────────── */}

      <main className="stagger-delay w-full max-w-[860px] px-6 py-12 sm:px-8">
        {/* Header strip */}
        <div className="mb-10 animate-fade-in-up border-b border-[#111] pb-3">
          <span className="font-mono text-xs tracking-[0.15em]" style={{ color: '#666' }}>
            RESEARCH PAPER | TRIADRANK-2026
          </span>
        </div>

        {/* ─── Mobile TOC ─────────────────────────────── */}
        <MobileToc />

        {/* ── 1. Overview ───────────────────────────────────── */}
        <section id="overview" className="mb-12 animate-fade-in-up scroll-mt-20">
          <h2 className="mb-4 text-2xl font-semibold">1. Overview</h2>
          <p className="mb-4 leading-relaxed">
            TriadRank is a three-tier resume scoring and ranking system designed for transparent, explainable
            applicant tracking. Unlike monolithic black-box approaches, TriadRank decomposes the scoring problem
            into three independently inspectable stages, each using a specialized model architecture.
          </p>
          <p className="leading-relaxed">
            The system processes resumes through a cascading pipeline: coarse filtering via entity extraction and
            keyword overlap, category validation via a DistilBERT classifier, and deep semantic scoring via a
            BERT-based cross-encoder. Each tier produces intermediate results that are surfaced through the
            interactive visualization layer.
          </p>
        </section>

        {/* ── 2. Architecture ──────────────────────────────── */}
        <section id="architecture" className="mb-12 animate-fade-in-up scroll-mt-20">
          <h2 className="mb-4 text-2xl font-semibold">2. Architecture</h2>
          <p className="mb-4 leading-relaxed">
            The pipeline is organized as a strict linear cascade. Each tier reduces the candidate set before
            passing results to the next, enabling efficient processing of large resume pools.
          </p>
          <div className="mb-4 border border-[#111] bg-[#F0F0EE] p-4 font-mono text-sm">
            score(candidate, JD) = f<sub>ce</sub>(candidate, JD) · P(category)
          </div>
          <p className="leading-relaxed">
            Where f<sub>ce</sub> is the cross-encoder regression score and P(category) is the category penalty
            factor (1.0 for matches, 0.5 for mismatches).
          </p>
        </section>

        {/* ── 3. Tier 3: Entity Extractor ──────────────────── */}
        <section id="tier3" className="mb-12 animate-fade-in-up scroll-mt-20">
          <h2 className="mb-4 text-2xl font-semibold">3. Tier 3: Entity Extractor</h2>
          <p className="mb-4 leading-relaxed">
            The first tier uses a fine-tuned spaCy NER model to extract structured entities from unstructured
            resume text. It identifies skills, experience, education, and personal information using a combination
            of pre-trained named entity recognition and custom pattern-matching rules.
          </p>

          <h3 className="mb-3 mt-8 text-xl font-semibold">Keyword Overlap (Jaccard)</h3>
          <p className="mb-3 leading-relaxed">
            A fast, interpretable baseline score computed as the Jaccard similarity between tokenized resume text
            and job description:
          </p>
          <div className="mb-4 border border-[#111] bg-[#F0F0EE] p-4 font-mono text-sm">
            J(A, B) = |A &#x2229; B| / |A &#x222A; B|
          </div>
          <CodeBlock
            code={`def keyword_overlap(resume: str, jd: str) -> float:
    tokens_r = set(tokenize(resume.lower()))
    tokens_j = set(tokenize(jd.lower()))
    if not tokens_r or not tokens_j:
        return 0.0
    return len(tokens_r & tokens_j) / len(tokens_r | tokens_j)`}
            filename="extractor.py"
          />

          <h3 className="mb-3 mt-8 text-xl font-semibold">Skill Overlap</h3>
          <p className="leading-relaxed">
            Beyond simple keyword matching, the extractor maintains a curated skill taxonomy and computes overlap
            between detected resume skills and required job description skills.
          </p>
        </section>

        {/* ── 4. Tier 2: Category Encoder ──────────────────── */}
        <section id="tier2" className="mb-12 animate-fade-in-up scroll-mt-20">
          <h2 className="mb-4 text-2xl font-semibold">4. Tier 2: Category Encoder</h2>
          <p className="mb-4 leading-relaxed">
            The second tier uses a DistilBERT-based classifier to predict the job category of each resume from 24
            predefined categories. The predicted category is compared to the target category; mismatches apply a
            configurable penalty to the final score.
          </p>
          <CodeBlock
            code={`class CategoryEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(config['pretrained_model'])
        self.classifier = nn.Linear(config['hidden_size'], config['num_classes'])

    def forward(self, input_ids, attention_mask):
        output = self.encoder(input_ids, attention_mask=attention_mask)
        pooled = output.last_hidden_state[:, 0, :]
        return self.classifier(pooled)`}
            filename="category_encoder.py"
          />
          <p className="mt-4 leading-relaxed">
            The category encoder achieves ~85% accuracy across the 24 categories. The penalty factor for
            mismatches is configurable via config.yaml (default: 0.5).
          </p>
        </section>

        {/* ── 5. Tier 1: Cross-Encoder ─────────────────────── */}
        <section id="tier1" className="mb-12 animate-fade-in-up scroll-mt-20">
          <h2 className="mb-4 text-2xl font-semibold">5. Tier 1: Cross-Encoder</h2>
          <p className="mb-3 leading-relaxed">
            The final scoring tier uses a BERT-base cross-encoder that jointly encodes the resume-job description
            pair and produces both a regression score (0–1) and a three-class classification label
            (Good Fit / Potential Fit / Bad Fit).
          </p>
          <p className="mb-3 leading-relaxed">
            The cross-encoder takes the concatenated resume and job description text (separated by a [SEP] token),
            processes it through the full transformer stack, and feeds the [CLS] representation to dual prediction
            heads:
          </p>
          <ul className="mb-4 list-disc space-y-1 pl-6">
            <li>Regression head: 768 &rarr; 256 &rarr; 1 (sigmoid)</li>
            <li>Classification head: 768 &rarr; 128 &rarr; 3 (softmax)</li>
          </ul>
          <CodeBlock
            code={`class MultiHeadCrossEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(config['pretrained_model'])
        self.regression_head = nn.Sequential(
            nn.Linear(config['hidden_size'], config['regression_head_units']),
            nn.ReLU(), nn.Dropout(config['dropout']),
            nn.Linear(config['regression_head_units'], 1),
            nn.Sigmoid()
        )
        self.classification_head = nn.Sequential(
            nn.Linear(config['hidden_size'], config['classification_head_units']),
            nn.ReLU(), nn.Dropout(config['dropout']),
            nn.Linear(config['classification_head_units'], config['num_classes'])
        )

    def forward(self, input_ids, attention_mask):
        output = self.encoder(input_ids, attention_mask=attention_mask)
        cls = output.last_hidden_state[:, 0, :]
        score = self.regression_head(cls).squeeze(-1)
        logits = self.classification_head(cls)
        return score, logits`}
            filename="cross_encoder.py"
          />
        </section>

        {/* ── 6. Scoring & Ranking ─────────────────────────── */}
        <section id="scoring" className="mb-12 animate-fade-in-up scroll-mt-20">
          <h2 className="mb-4 text-2xl font-semibold">6. Scoring &amp; Ranking</h2>
          <p className="mb-3 leading-relaxed">
            The final score for each candidate is computed as the cross-encoder regression output multiplied by the
            category penalty factor:
          </p>
          <div className="mb-4 border border-[#111] bg-[#F0F0EE] p-4 font-mono text-sm">
            score<sub>final</sub> = score<sub>ce</sub> &times; penalty(category<sub>pred</sub>, category<sub>target</sub>)
          </div>
          <p className="leading-relaxed">
            Candidates are sorted by descending final score. The pipeline evaluates all candidates through Tier 3,
            passes the top-K to Tier 2, and the surviving candidates to Tier 1. This tiered filtering ensures O(n)
            scaling with candidate pool size.
          </p>
        </section>

        {/* ── 7. Bias Mitigation ─────────────────────────────── */}
        <section id="bias" className="mb-12 animate-fade-in-up scroll-mt-20">
          <h2 className="mb-4 text-2xl font-semibold">7. Bias Mitigation</h2>
          <p className="mb-4 leading-relaxed">TriadRank incorporates several bias mitigation strategies:</p>
          <ul className="list-disc space-y-2 pl-6">
            <li>
              <strong>Text normalization:</strong> Strips HTML, normalizes whitespace, removes PII-like patterns
            </li>
            <li>
              <strong>Category penalty transparency:</strong> Mismatch penalties are explicit and configurable
            </li>
            <li>
              <strong>Score decomposition:</strong> Every score component is visualized individually
            </li>
            <li>
              <strong>Threshold reporting:</strong> Good-fit &ge; 0.75, potential-fit &ge; 0.5 — documented and
              surfaced
            </li>
          </ul>
        </section>

        {/* ── 8. References ──────────────────────────────────── */}
        <section id="references" className="mb-12 animate-fade-in-up scroll-mt-20">
          <h2 className="mb-4 text-2xl font-semibold">8. References</h2>
          <ul className="space-y-3">
            <li className="leading-relaxed">
              Devlin, J. et al. (2019). &ldquo;BERT: Pre-training of Deep Bidirectional Transformers for Language
              Understanding.&rdquo; <em>NAACL</em>.
            </li>
            <li className="leading-relaxed">
              Sanh, V. et al. (2019). &ldquo;DistilBERT, a distilled version of BERT: smaller, faster, cheaper and
              lighter.&rdquo; <em>NeurIPS Workshop</em>.
            </li>
            <li className="leading-relaxed">
              Honnibal, M. &amp; Montani, I. (2017). &ldquo;spaCy 2: Natural language understanding with Bloom
              embeddings&hellip;&rdquo; <em>EMNLP</em>.
            </li>
            <li className="leading-relaxed">
              Wolf, T. et al. (2020). &ldquo;Transformers: State-of-the-Art Natural Language Processing.&rdquo;
              <em>EMNLP</em>.
            </li>
          </ul>
        </section>

        {/* ponytail: page-level metadata such as DOI, citation footer, or print stylesheet link can go here if needed */}
      </main>
    </div>
  )
}
