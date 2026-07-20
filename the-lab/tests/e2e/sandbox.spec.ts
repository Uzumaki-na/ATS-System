import { test, expect } from "@playwright/test";
import { resolve } from "node:path";

/* ─────────────────────────────────────────────────────────────────────────
 * Phase B — end-to-end integration + OUTPUT-CORRECTNESS gate.
 *
 * Fixture = datasets/resume_pdfs/INFORMATION-TECHNOLOGY/10089434.pdf — an
 * IT SYSTEMS ADMINISTRATOR (Active Directory / Azure / VMware / backup
 * infra, BSc Information Technology 2005). The sandbox's default JD is the
 * "Senior Machine Learning Engineer" sample (category ENGINEERING), so this
 * is an UNAMBIGUOUS MISMATCH: a sysadmin is not a good fit for a senior ML
 * role, and the resume contains none of the ML stack. That sharp contrast is
 * the golden ground truth — the gate exists to catch a regression that makes
 * it look correct while the underlying values are wrong.
 *
 * Golden ground truth (read from the resume itself — the backend sees the same
 * text via pdfplumber at x/y_tolerance=2):
 *   - true role domain       : IT systems admin          → predict INFORMATION-TECHNOLOGY
 *   - JD-skill hallucination : NONE (resume has no ML stack) → expect NONE
 *   - fit verdict            : mismatch                   → expect Bad Fit, low score
 *   - experience             : IT work since Aug 2005      → expect years ∈ (0,60)
 *   - education              : BSc Information Technology  → expect "Bachelor"-bearing entry
 *
 * Bar (decided with user): HARD-FAIL gross regressions, WARN subtle known
 * upstream Tier-3 extractor limitations (SkillTrie substring false-positives,
 * education-section contamination, weighted-blend inversion). Warns log to
 * the run output; they DO NOT fail the lane. Raw payload is read from
 * sessionStorage["sandbox_results"] — the authoritative extractor output
 * BEFORE view-model mapping — so the gate inspects the real backend layer,
 * not just the rendered layer, and checks the two agree.
 * ───────────────────────────────────────────────────────────────────────── */

const FIXTURE = resolve(
  process.cwd(),
  "..",
  "datasets",
  "resume_pdfs",
  "INFORMATION-TECHNOLOGY",
  "10089434.pdf",
);

// Skills the resume provably does NOT contain — the extractor must not
// hallucinate the JD's stack onto the resume. (Any hit = gross corruption.)
const FORBIDDEN_HALLUCINATION = [
  "Python", "PyTorch", "TensorFlow", "scikit-learn",
  "Kubernetes", "Docker", "MLflow",
];

// Known SkillTrie Aho-Corasick substring false-positives observed on THIS
// resume (extractor-quality limitation, NOT an integration regression) —
// warn only. These arise from word-prefix/substring matches in the trie.
const KNOWN_SUBSTRING_JUNK = [
  "Lysis", "Syman", "Monit", "STATISTICA", "LESS", "NDepend", "Tadoma",
];

// Categories the backend does NOT have — the old mock-data.ts shipped these
// 16 dead names; the runtime /categories fetch must surface only the real 24.
const DEAD_CATEGORY_NAMES = [
  "MARKETING", "EDUCATION", "LEGAL", "REAL-ESTATE", "RESTAURANT",
  "TRANSPORTATION", "ADMINISTRATIVE", "ARCHITECTURE", "CREATIVE-ARTS",
  "CUSTOMER-SERVICE",
];

test.describe("sandbox → results: integration + output correctness", () => {
  test.beforeEach(async ({ request }) => {
    // Full-stack precondition: Docker backend reachable THROUGH the proxy.
    // Skip cleanly if down — a missing stack must not masquerade as a code bug.
    let healthy = false;
    try {
      const r = await request.get("/api/health", { timeout: 8_000 });
      healthy = r.ok();
    } catch {
      healthy = false;
    }
    test.skip(
      !healthy,
      "Backend unhealthy at /api/health — start the Docker backend on :8000 before running e2e.",
    );
  });

  test("sysadmin resume vs Senior-ML-Engineer JD → correct mismatch verdict + sane outputs", async ({ page }) => {
    const warnings: string[] = [];

    /* ── 1. Sandbox loads ─────────────────────────────────────────────── */
    await page.goto("/sandbox");
    const initBtn = page.getByRole("button", { name: /Initialize Inference/i });
    await expect(initBtn).toBeVisible();

    /* ── 2. Runtime categories: 24 real, incl. IT, excl. the dead names ── */
    await page.getByRole("combobox").click();
    const options = page.getByRole("option");
    await expect(options.first()).toBeVisible();
    const optTexts = (await options.allTextContents()).map((s) => s.trim());
    expect(optTexts.length, "backend exposes exactly 24 categories").toBe(24);
    expect(optTexts).toContain("INFORMATION-TECHNOLOGY");
    for (const dead of DEAD_CATEGORY_NAMES) {
      expect(optTexts, `dead mock category must not be offered: ${dead}`).not.toContain(dead);
    }
    await page.keyboard.press("Escape"); // close dropdown; default ENGINEERING stays

    /* ── 3. JD is the Senior ML Engineer sample (the mismatch premise) ─── */
    const jd = page.locator("textarea").first();
    await expect(jd).not.toBeEmpty();
    await expect(jd).toHaveValue(/Machine Learning/i);

    /* ── 4. Upload the fixture via the (hidden) file input ─────────────── */
    await page.locator('input[type="file"]').setInputFiles(FIXTURE);
    await expect(page.getByText("10089434.pdf")).toBeVisible();

    /* ── 5. Run inference → land on /results ───────────────────────────── */
    await initBtn.click();
    await page.waitForURL("**/results**", { timeout: 60_000 });

    /* ── 6. Read the authoritative raw payload from sessionStorage ─────── */
    const raw = await page.evaluate(() => {
      const s = sessionStorage.getItem("sandbox_results");
      return s ? JSON.parse(s) : null;
    });
    expect(raw, "sandbox_results must be written before navigating to /results").toBeTruthy();
    expect(raw.results.length, "at least one candidate returned").toBeGreaterThanOrEqual(1);
    const cand = raw.results[0];

    /* ═════════════════════════════════════════════════════════════════════
     *  RAW OUTPUT CORRECTNESS — the "is the value actually right" gate
     * ═════════════════════════════════════════════════════════════════════ */

    expect(cand.candidate_id, "candidate_id is the uploaded filename (API contract)").toBe("10089434.pdf");

    // (a) Mismatch verdict — a sysadmin must NOT be a Good Fit for Sr ML Eng.
    expect(["Good Fit", "Potential Fit", "Bad Fit"]).toContain(cand.label);
    expect(cand.label, "sysadmin vs Sr ML Eng must not read 'Good Fit'").not.toBe("Good Fit");
    expect(cand.final_score, "final_score capped to [0,1], low for mismatch").toBeLessThan(0.5);

    // (b) Label probabilities: valid range, sum ~1, NOT uniform (model not collapsed).
    const lp = cand.label_probabilities;
    for (const v of Object.values(lp) as number[]) {
      expect(v).toBeGreaterThanOrEqual(0);
      expect(v).toBeLessThanOrEqual(1);
    }
    expect(lp.good_fit + lp.potential_fit + lp.bad_fit).toBeCloseTo(1, 2);
    expect(lp.bad_fit, "mismatch candidate must not produce uniform/degenerate probs").toBeGreaterThan(
      lp.good_fit,
    );

    // (c) Category encoder accurate on this fixture — predicted IT, high confidence.
    expect(cand.category.predicted, "sysadmin resume → predicted IT").toBe("INFORMATION-TECHNOLOGY");
    expect(cand.category.confidence, "category confidence high on a prototypical IT resume").toBeGreaterThan(0.9);
    expect(cand.category.match, "ENGINEERING JD ≠ IT resume → match false").toBe(false);

    // (d) Overlap + score bounds.
    for (const k of ["skill_overlap", "keyword_overlap", "final_score", "raw_score"]) {
      expect(cand[k], `${k} ∈ [0,1]`).toBeGreaterThanOrEqual(0);
      expect(cand[k], `${k} ∈ [0,1]`).toBeLessThanOrEqual(1);
    }

    // (e) Experience + processing time real, not degenerate defaults.
    expect(cand.experience_years, "years > 0 (Aug-2005 start)").toBeGreaterThan(0);
    expect(cand.experience_years, "years < 60 (sanity ceiling)").toBeLessThan(60);
    expect(cand.processing_time, "processing_time > 0").toBeGreaterThan(0);

    // (f) Education non-empty and "Bachelor"-bearing (it IS a BSc IT).
    expect(cand.education.length, "education entries returned").toBeGreaterThan(0);
    expect(
      cand.education.some((e: string) => /bachelor/i.test(e)),
      "at least one education entry mentions a Bachelor's degree",
    ).toBe(true);

    // (g) Skills: many surfaced, well-formed, NO JD-stack hallucination.
    expect(cand.skills.length, "SkillTrie should surface many skills on a verbose resume").toBeGreaterThan(20);
    for (const s of cand.skills) {
      expect(s.trim().length, `no degenerate/whitespace skill token: ${JSON.stringify(s)}`).toBeGreaterThanOrEqual(2);
    }
    const lower = cand.skills.map((s: string) => s.toLowerCase());
    for (const h of FORBIDDEN_HALLUCINATION) {
      expect(
        lower,
        `must NOT hallucinate a JD skill the resume provably lacks: ${h}`,
      ).not.toContain(h.toLowerCase());
    }

    /* ═════════════════════════════════════════════════════════════════════
     *  RENDERED CONSISTENCY — table displays the raw values (no drift)
     * ═════════════════════════════════════════════════════════════════════ */
    const dispScore = (Math.round(cand.final_score * 10_000) / 10_000).toFixed(4);
    const dispSO = Math.round(Math.min(1, Math.max(0, cand.skill_overlap)) * 100);
    const dispExp = Math.round(cand.experience_years);
    const dispMs = Math.round(cand.processing_time * 1000);

    const row = page.locator("table tbody tr").first();
    await expect(row).toBeVisible();
    await expect(row.locator("td").nth(1)).toContainText(dispScore);
    await expect(row.locator("td").nth(2)).toContainText(cand.label);
    await expect(row.locator("td").nth(3)).toContainText("INFORMATION-TECHNOLOGY");
    await expect(row.locator("td").nth(4)).toContainText(/✗/); // ✗ for mismatch
    await expect(row.locator("td").nth(5)).toContainText(`${dispSO}%`);
    await expect(row.locator("td").nth(6)).toContainText(`${dispExp}y`);
    await expect(row.locator("td").nth(7)).toContainText(`${dispMs}ms`);

    /* ═════════════════════════════════════════════════════════════════════
     *  DETAIL MODAL — same numbers + penalty + extraction detail
     * ═════════════════════════════════════════════════════════════════════ */
    await row.click();
    const dlg = page.getByRole("dialog");
    await expect(dlg.getByText(/Candidate Detail/i)).toBeVisible();
    await expect(dlg.getByText(dispScore)).toBeVisible();
    // The label string renders twice in the modal — as the colored verdict
    // pill (LabelBadge, span.inline-block.font-mono.font-semibold) AND as a
    // bare <span> in the static "Label Probabilities" legend. Scope to the
    // pill's class signature so we assert the authoritative verdict render,
    // not the always-present legend row. (Score span lacks inline-block;
    // skill chips lack font-semibold → pill is the unique triple match.)
    const verdictPill = dlg.locator(
      "span.inline-block.font-mono.font-semibold",
      { hasText: cand.label },
    );
    await expect(verdictPill).toBeVisible();
    await expect(dlg.getByText("INFORMATION-TECHNOLOGY")).toBeVisible();
    await expect(dlg.getByText(/✗\s*Mismatched/)).toBeVisible(); // ✓/✗ Mismatched
    await expect(dlg.getByText("Category mismatch")).toBeVisible();
    await expect(dlg.getByText(/x0\.50/)).toBeVisible();
    await expect(dlg.getByText(`Extracted Skills (${cand.skills.length})`)).toBeVisible();

    /* ═════════════════════════════════════════════════════════════════════
     *  WARNINGS — known upstream extractor limitations (log, don't fail)
     * ═════════════════════════════════════════════════════════════════════ */
    const junkFound = cand.skills.filter((s: string) => KNOWN_SUBSTRING_JUNK.includes(s));
    if (junkFound.length) {
      warnings.push(
        `SkillTrie substring false-positives (extractor limitation, NOT integration bug): ${junkFound.join(", ")}`,
      );
    }
    if (cand.education[0] && !/bachelor/i.test(cand.education[0])) {
      warnings.push(
        `education[0] is a job-duty blob, not the degree (extractor education-segmentation weakness): "${String(cand.education[0]).slice(0, 60)}…"`,
      );
    }
    if (cand.final_score > cand.raw_score + 1e-9) {
      warnings.push(
        `final_score ${cand.final_score} > raw_score ${cand.raw_score} despite a category-mismatch penalty (weighted-blend inversion — investigate separately)`,
      );
    }
    if (warnings.length) {
      console.log(
        "\n[e2e WARN — known upstream extractor limitations, NOT failures]\n" +
          warnings.map((w) => "  • " + w).join("\n") +
          "\n",
      );
    }
  });
});
