/* ─── API Client for TriadRank Backend ─────────────────────── */

// Same-origin /api/* is proxied to the backend by next.config.ts rewrites
// (server-side, so the backend URL never ships to the browser). Set
// BACKEND_URL on the Next server to override the default http://localhost:8000.
const API_BASE = "/api";

/* ─── API response types (snake_case, matches backend) ────── */

export interface ApiLabelProbability {
  good_fit: number;
  potential_fit: number;
  bad_fit: number;
}

export interface ApiCategoryResult {
  predicted: string;
  match: boolean;
  confidence: number;
}

export interface ApiCandidateOutput {
  rank: number;
  candidate_id: string;
  final_score: number;
  raw_score: number;
  label: string;
  label_probabilities: ApiLabelProbability;
  category: ApiCategoryResult;
  skill_overlap: number;
  keyword_overlap: number;
  skills: string[];
  experience_years: number;
  education: string[];
  processing_time: number;
}

export interface ApiRankResponse {
  job_id: string;
  job_category: string;
  total_candidates: number;
  returned_candidates: number;
  processing_time_seconds: number;
  results: ApiCandidateOutput[];
}

/* ─── Errors ────────────────────────────────────────────────── */

export class ApiError extends Error {
  status: number;
  detail: string;
  constructor(status: number, detail: string) {
    super(`API error ${status}: ${detail}`);
    this.name = "ApiError";
    this.status = status;
    this.detail = detail;
  }
}

/* ─── Endpoints ─────────────────────────────────────────────── */

/** Quick connectivity check — returns false instead of throwing. */
export async function healthCheck(): Promise<boolean> {
  try {
    const res = await fetch(`${API_BASE}/health`, { signal: AbortSignal.timeout(5000) });
    return res.ok;
  } catch {
    return false;
  }
}

/**
 * POST /rank/pdf — upload PDF resumes and run the pipeline.
 * Throws ApiError on failure.
 */
export async function rankPdfs(
  jobDescription: string,
  jobCategory: string,
  files: File[],
): Promise<ApiRankResponse> {
  const formData = new FormData();
  formData.append("job_description", jobDescription);
  formData.append("job_category", jobCategory);

  for (const file of files) {
    formData.append("files", file);
  }

  const res = await fetch(`${API_BASE}/rank/pdf`, {
    method: "POST",
    body: formData,
  });

  if (!res.ok) {
    let detail = `HTTP ${res.status}`;
    try {
      const body = await res.json();
      if (body?.detail) detail = body.detail;
    } catch {
      /* unable to parse body — use default */
    }
    throw new ApiError(res.status, detail);
  }

  return res.json();
}

/** GET /categories — list all supported categories. */
export async function fetchCategories(): Promise<string[]> {
  const res = await fetch(`${API_BASE}/categories`);
  if (!res.ok) throw new ApiError(res.status, "Failed to fetch categories");
  const data = await res.json();
  return data.categories ?? [];
}
