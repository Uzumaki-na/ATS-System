/**
 * API → view-model mapper + the shared display types consumed by
 * results/page.tsx and the detail modal. No mock data lives here
 * (the seeded mock generator was removed when the backend was wired).
 */

import type { ApiCandidateOutput } from "@/lib/api";

/* ─── Display types (shared with detail modal) ──────────── */

export interface EntityExtractionItem {
  jd: string;
  resume: string;
}

export interface PenaltyItem {
  reason: string;
  multiplier: number;
}

export interface LabelProbabilities {
  goodFit: number;
  potentialFit: number;
  badFit: number;
}

export interface Candidate {
  id: string;
  name: string;
  score: number;
  rawScore: number;
  label: "Good Fit" | "Potential Fit" | "Bad Fit";
  category: string;
  categoryMatch: boolean;
  categoryConfidence: number;
  skills: string[];
  skillOverlap: number;
  keywordOverlap: number;
  experienceYears: number;
  educationLevel: "Bachelor's" | "Master's" | "PhD" | "Associate's";
  entityExtractionChain: EntityExtractionItem[];
  processingTimeMs: number;
  semantic: number;
  syntax: number;
  experience: number;
  education: number;
  keywords: number;
  penalties: PenaltyItem[];
  labelProbabilities: LabelProbabilities;
}

/* ─── API Response → View Model Mapper ──────────────────── */

function _cleanCandidateId(id: string): string {
  // "resume_john_doe.pdf" → "John Doe"
  // "CAND-001" → "Candidate 001"
  const stripped = id.replace(/\.(pdf|txt)$/i, "");
  const cleaned = stripped
    .replace(/[_-]+/g, " ")
    .replace(/\b\w/g, (c) => c.toUpperCase())
    .trim();
  // If still all-caps (e.g. "CAND-001" → "Cand 001") flag it
  if (/^[A-Z\s]+$/.test(cleaned)) return `Candidate ${cleaned.replace(/\s+/g, " ")}`;
  return cleaned;
}

function _deriveEducationLevel(education: string[]): Candidate["educationLevel"] {
  const text = education.join(" ").toLowerCase();
  if (/phd|doctorate/.test(text)) return "PhD";
  if (/master/.test(text)) return "Master's";
  if (/bachelor/.test(text)) return "Bachelor's";
  if (/associate/.test(text)) return "Associate's";
  return education.length > 0 ? "Bachelor's" : "Associate's";
}

/**
 * Convert a backend ApiCandidateOutput (snake_case) into a frontend
 * Candidate (camelCase), deriving sensible values for the radar chart
 * and other display-only fields the backend doesn't compute directly.
 *
 * Derived radar axes:
 *   semantic  ← label_probabilities.good_fit  (confidence it's a fit)
 *   syntax    ← keyword_overlap               (textual keyword match)
 *   experience ← experience_years / 15        (normalised to 0-1 cap)
 *   education  ← 0.8 if education list non-empty, else 0.2
 *   keywords   ← skill_overlap                (skill match)
 *
 * Penalties only reflect what the backend applies (category mismatch).
 * Entity "chain" is a simplified summary of what was extracted.
 */
export function apiCandidateToView(api: ApiCandidateOutput): Candidate {
  const skillPct = Math.round(Math.min(1, Math.max(0, api.skill_overlap)) * 100);
  const keywordPct = Math.round(Math.min(1, Math.max(0, api.keyword_overlap)) * 100);

  /* ── Radar sub-scores ── */
  const semantic = Math.round(Math.min(1, Math.max(0, api.label_probabilities.good_fit)) * 10000) / 10000;
  const syntax = Math.round(Math.min(1, Math.max(0, api.keyword_overlap)) * 10000) / 10000;
  const expScore = Math.round(Math.min(1, Math.max(0, api.experience_years / 15)) * 10000) / 10000;
  const eduScore = api.education.length > 0 ? 0.8 : 0.2;
  const kwScore = Math.round(Math.min(1, Math.max(0, api.skill_overlap)) * 10000) / 10000;

  /* ── Penalties ── */
  const penalties: PenaltyItem[] = [];
  if (!api.category.match) {
    penalties.push({ reason: "Category mismatch", multiplier: 0.5 });
  }

  /* ── Entity info (simplified summary) ── */
  const entityInfo: EntityExtractionItem[] = [];
  if (api.skills.length > 0) {
    entityInfo.push({
      jd: "Required skills",
      resume: api.skills.slice(0, 6).join(", "),
    });
  }
  if (api.experience_years > 0) {
    entityInfo.push({
      jd: `${Math.ceil(api.experience_years)}+ years experience`,
      resume: `${api.experience_years} years`,
    });
  }
  if (api.education.length > 0) {
    entityInfo.push({
      jd: "Education requirements",
      resume: api.education[0],
    });
  }

  return {
    id: api.candidate_id,
    name: _cleanCandidateId(api.candidate_id),
    score: Math.round(api.final_score * 10000) / 10000,
    rawScore: Math.round(api.raw_score * 10000) / 10000,
    label: api.label as Candidate["label"],
    category: api.category.predicted,
    categoryMatch: api.category.match,
    categoryConfidence: Math.round(api.category.confidence * 10000) / 10000,
    skills: api.skills,
    skillOverlap: skillPct,
    keywordOverlap: keywordPct,
    experienceYears: Math.round(api.experience_years),
    educationLevel: _deriveEducationLevel(api.education),
    entityExtractionChain: entityInfo,
    processingTimeMs: Math.round(api.processing_time * 1000),
    semantic,
    syntax,
    experience: expScore,
    education: eduScore,
    keywords: kwScore,
    penalties,
    labelProbabilities: {
      goodFit: api.label_probabilities.good_fit,
      potentialFit: api.label_probabilities.potential_fit,
      badFit: api.label_probabilities.bad_fit,
    },
  };
}
