import { describe, it, expect } from "vitest";
import { apiCandidateToView } from "../../lib/view-models";
import type { ApiCandidateOutput } from "../../lib/api";

/* Realistic snake_case candidate (mirrors the accountant live-proof:
   raw 0.618 Good Fit, skill_overlap 0.571, category match). */
const baseApi: ApiCandidateOutput = {
  rank: 1,
  candidate_id: "resume_jane_smith.pdf",
  final_score: 0.618,
  raw_score: 0.618,
  label: "Good Fit",
  label_probabilities: { good_fit: 0.91, potential_fit: 0.06, bad_fit: 0.03 },
  category: { predicted: "ACCOUNTANT", match: true, confidence: 0.88 },
  skill_overlap: 0.571,
  keyword_overlap: 0.419,
  skills: ["Excel", "QuickBooks", "GAAP", "Reconciliation"],
  experience_years: 7,
  education: ["Bachelor of Science in Accounting"],
  processing_time: 0.42,
};

describe("apiCandidateToView — core mapping (snake → camel)", () => {
  it("copies identifiable + label fields verbatim", () => {
    const v = apiCandidateToView(baseApi);
    expect(v.id).toBe("resume_jane_smith.pdf");
    expect(v.label).toBe("Good Fit");
    expect(v.category).toBe("ACCOUNTANT");
    expect(v.categoryMatch).toBe(true);
    expect(v.skills).toEqual(["Excel", "QuickBooks", "GAAP", "Reconciliation"]);
  });

  it("rounds scores to 4 decimals", () => {
    const v = apiCandidateToView({ ...baseApi, final_score: 0.618123, raw_score: 0.555999 });
    expect(v.score).toBe(0.6181);
    expect(v.rawScore).toBe(0.556);
  });

  it("converts processing_time (seconds) → ms (rounded)", () => {
    expect(apiCandidateToView({ ...baseApi, processing_time: 0.42 }).processingTimeMs).toBe(420);
    expect(apiCandidateToView({ ...baseApi, processing_time: 1.0049 }).processingTimeMs).toBe(1005);
  });

  it("rounds experience years (banker's rounding to int)", () => {
    expect(apiCandidateToView({ ...baseApi, experience_years: 7.6 }).experienceYears).toBe(8);
    expect(apiCandidateToView({ ...baseApi, experience_years: 7.4 }).experienceYears).toBe(7);
  });

  it("renames label_probabilities → labelProbabilities", () => {
    expect(apiCandidateToView(baseApi).labelProbabilities).toEqual({
      goodFit: 0.91,
      potentialFit: 0.06,
      badFit: 0.03,
    });
  });
});

describe("apiCandidateToView — radar axes (derived, clamped to [0,1])", () => {
  it("semantic ← good_fit, syntax ← keyword_overlap, keywords ← skill_overlap", () => {
    const v = apiCandidateToView(baseApi);
    expect(v.semantic).toBe(0.91);
    expect(v.syntax).toBe(0.419);
    expect(v.keywords).toBe(0.571);
  });

  it("experience ← years/15, capped at 1.0", () => {
    expect(apiCandidateToView({ ...baseApi, experience_years: 7 }).experience).toBeCloseTo(0.4667, 3);
    expect(apiCandidateToView({ ...baseApi, experience_years: 30 }).experience).toBe(1); // 30/15=2 → clamped
  });

  it("education ← 0.8 if education non-empty, else 0.2", () => {
    expect(apiCandidateToView(baseApi).education).toBe(0.8);
    expect(apiCandidateToView({ ...baseApi, education: [] }).education).toBe(0.2);
  });

  it("clamps negative / >1 overlaps before scaling (skillOverlap & keywordOverlap are % ints)", () => {
    const v = apiCandidateToView({ ...baseApi, skill_overlap: 1.5, keyword_overlap: -0.3 });
    expect(v.skillOverlap).toBe(100); // 1.5 → 1.0 → 100
    expect(v.keywordOverlap).toBe(0); // -0.3 → 0 → 0
    expect(v.keywords).toBe(1); // raw-ish axis keeps 0-1
    expect(v.syntax).toBe(0);
  });
});

describe("apiCandidateToView — penalties", () => {
  it("no penalty when category matches", () => {
    expect(apiCandidateToView(baseApi).penalties).toEqual([]);
  });

  it("adds a Category-mismatch penalty (0.5×) when !category.match", () => {
    const v = apiCandidateToView({
      ...baseApi,
      category: { predicted: "ADVOCATE", match: false, confidence: 0.7 },
    });
    expect(v.penalties).toEqual([{ reason: "Category mismatch", multiplier: 0.5 }]);
    expect(v.categoryMatch).toBe(false);
  });
});

describe("apiCandidateToView — entity-extraction summary", () => {
  it("summarises skills (≤6), experience (ceil →.jd), education[0]", () => {
    const v = apiCandidateToView({ ...baseApi, experience_years: 7.4 });
    expect(v.entityExtractionChain).toEqual([
      { jd: "Required skills", resume: "Excel, QuickBooks, GAAP, Reconciliation" },
      { jd: "8+ years experience", resume: "7.4 years" }, // Math.ceil(7.4)=8
      { jd: "Education requirements", resume: "Bachelor of Science in Accounting" },
    ]);
  });

  it("caps displayed skills at 6 (slice)", () => {
    const eight = ["a", "b", "c", "d", "e", "f", "g", "h"];
    const v = apiCandidateToView({ ...baseApi, skills: eight });
    const skillsEntry = v.entityExtractionChain.find((e) => e.jd === "Required skills");
    expect(skillsEntry?.resume).toBe("a, b, c, d, e, f");
  });

  it("omits an entry when its source is empty/zero", () => {
    const v = apiCandidateToView({ ...baseApi, skills: [], experience_years: 0, education: [] });
    expect(v.entityExtractionChain).toEqual([]);
  });
});

describe("apiCandidateToView — education-level derivation", () => {
  it("PhD / Doctorate → 'PhD'", () => {
    expect(apiCandidateToView({ ...baseApi, education: ["PhD in CS"] }).educationLevel).toBe("PhD");
    expect(apiCandidateToView({ ...baseApi, education: ["Doctorate of Eng"] }).educationLevel).toBe("PhD");
  });
  it("Master* → 'Master's'", () => {
    expect(apiCandidateToView({ ...baseApi, education: ["Master of Science"] }).educationLevel).toBe("Master's");
  });
  it("Bachelor* → 'Bachelor's'", () => {
    expect(apiCandidateToView({ ...baseApi, education: ["Bachelor of Arts"] }).educationLevel).toBe("Bachelor's");
  });
  it("Associate* → 'Associate's'", () => {
    expect(apiCandidateToView({ ...baseApi, education: ["Associate Degree"] }).educationLevel).toBe("Associate's");
  });
  it("non-empty but no keyword (e.g. 'BSc') → default 'Bachelor's'", () => {
    expect(apiCandidateToView({ ...baseApi, education: ["BSc Computer Science"] }).educationLevel).toBe("Bachelor's");
  });
  it("empty → 'Associate's'", () => {
    expect(apiCandidateToView({ ...baseApi, education: [] }).educationLevel).toBe("Associate's");
  });
});

describe("apiCandidateToView — candidate-name cleaning (via candidate_id)", () => {
  it("resume_john_doe.pdf → 'Resume John Doe'", () => {
    expect(apiCandidateToView({ ...baseApi, candidate_id: "resume_john_doe.pdf" }).name).toBe("Resume John Doe");
  });
  it("strips .txt extension + titlecases word-initial letters", () => {
    expect(apiCandidateToView({ ...baseApi, candidate_id: "cv_jane_smith.txt" }).name).toBe("Cv Jane Smith");
  });
  // FLAGGED (rule #3 surface-don't-touch): the docstring claims
  // "CAND-001" → "Candidate 001", but the all-caps branch can never fire for
  // IDs containing digits (digits break /^[A-Z\s]+$/) and the titlecase step
  // never lowercases internal letters, so "CAND" stays uppercase. Asserting
  // ACTUAL behavior as living docs; docstring/logic mismatch reported to user.
  it("CAND-001 → 'CAND 001' (NOT the documented 'Candidate 001')", () => {
    expect(apiCandidateToView({ ...baseApi, candidate_id: "CAND-001" }).name).toBe("CAND 001");
  });
  it("pure-alpha uppercase id reaches the 'Candidate …' branch", () => {
    expect(apiCandidateToView({ ...baseApi, candidate_id: "ABC-DEF" }).name).toBe("Candidate ABC DEF");
  });
});
