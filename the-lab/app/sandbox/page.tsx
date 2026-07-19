"use client";

import { useState, useRef, useCallback, useEffect, type ChangeEvent, type DragEvent } from "react";
import { useRouter } from "next/navigation";
import { rankPdfs, ApiError, fetchCategories } from "@/lib/api";
import Terminal from "@/components/terminal";
import {
  Select,
  SelectTrigger,
  SelectContent,
  SelectItem,
  SelectValue,
} from "@/components/ui/select";
import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";

const SAMPLE_JD = `Senior Machine Learning Engineer

We are looking for a Senior Machine Learning Engineer to join our AI team. You will be responsible for designing, building, and deploying production-grade ML systems that power our core product.

Responsibilities:
- Design and implement end-to-end ML pipelines for training and serving models
- Build scalable data processing pipelines using Python, PySpark, and SQL
- Deploy and monitor models in production using Docker, Kubernetes, and cloud platforms (AWS/GCP)
- Collaborate with product and engineering teams to define ML requirements
- Conduct A/B experiments and statistical analysis to validate model improvements
- Mentor junior engineers and contribute to code reviews

Requirements:
- 5+ years of experience in software engineering with at least 3 years in ML
- Strong proficiency in Python and ML frameworks (PyTorch, TensorFlow, scikit-learn)
- Experience with NLP techniques and transformer architectures (BERT, GPT, etc.)
- Solid understanding of deep learning fundamentals and model optimization
- Experience with feature stores, model registries, and ML CI/CD
- Strong communication skills and ability to work in cross-functional teams

Nice to Have:
- Experience with LLM fine-tuning and prompt engineering
- Published research at top-tier ML conferences
- Experience with graph neural networks or recommendation systems
- Knowledge of MLOps best practices and tools (MLflow, Weights & Biases)

Education:
- MS or PhD in Computer Science, Machine Learning, or related field`;

export default function SandboxPage() {
  const router = useRouter();
  const fileInputRef = useRef<HTMLInputElement>(null);
  const [jdText, setJdText] = useState(SAMPLE_JD);
  const [files, setFiles] = useState<File[]>([]);
  const [category, setCategory] = useState("ENGINEERING");
  // Runtime categories from GET /categories (backend is the single source of
  // truth — no stale hardcoded list). Falls back to ["ENGINEERING"] on failure
  // so submission still works (ENGINEERING is a valid backend category).
  const [categories, setCategories] = useState<string[]>([]);
  const [isProcessing, setIsProcessing] = useState(false);
  const [isDragging, setIsDragging] = useState(false);
  const [logLines, setLogLines] = useState<string[]>([]);
  const [error, setError] = useState("");

  /* ─── Persist to sessionStorage ─────────────────────────────── */

  useEffect(() => {
    try { sessionStorage.setItem("sandbox_jd", jdText); } catch { /* noop */ }
  }, [jdText]);

  useEffect(() => {
    try { sessionStorage.setItem("sandbox_category", category); } catch { /* noop */ }
  }, [category]);

  // Fetch the real 24 categories at runtime (hydration-safe: only runs
  // client-side after mount; SSR + first paint both render the loading state).
  useEffect(() => {
    fetchCategories()
      .then(setCategories)
      .catch(() => setCategories(["ENGINEERING"]));
  }, []);

  /* ─── Handlers ──────────────────────────────────────────────── */

  const handleFilesAdded = useCallback((incoming: FileList | File[]) => {
    const MAX_FILES = 20;
    const allowed = [".pdf"];
    const incomingArr = Array.from(incoming);

    const valid = incomingArr.filter((f) =>
      allowed.some((ext) => f.name.toLowerCase().endsWith(ext)),
    );
    const rejected = incomingArr.length - valid.length;
    if (rejected > 0) {
      setError(`Only .pdf files are accepted. ${rejected} file(s) skipped.`);
    }

    setFiles((prev) => {
      const combined = [...prev, ...valid];
      const seen = new Set<string>();
      const deduped = combined.filter((f) => {
        const key = `${f.name}:${f.size}`;
        if (seen.has(key)) return false;
        seen.add(key);
        return true;
      });
      if (deduped.length > MAX_FILES) {
        setError(`Maximum ${MAX_FILES} files. Extra files were ignored.`);
      }
      return deduped.slice(0, MAX_FILES);
    });
  }, []);

  const handleDrop = useCallback((e: DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    setIsDragging(false);
    if (e.dataTransfer.files.length > 0) handleFilesAdded(e.dataTransfer.files);
  }, [handleFilesAdded]);

  const handleDragOver = useCallback((e: DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    setIsDragging(true);
  }, []);

  const handleDragLeave = useCallback((e: DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    setIsDragging(false);
  }, []);

  const handleFileSelect = useCallback((e: ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files.length > 0) {
      handleFilesAdded(e.target.files);
      e.target.value = "";
    }
  }, [handleFilesAdded]);

  const handleReset = useCallback(() => {
    if ((files.length > 0 || jdText.trim()) && !window.confirm("Clear all inputs? This cannot be undone.")) {
      return;
    }
    setJdText("");
    setFiles([]);
    setCategory("ENGINEERING");
    setLogLines([]);
    setError("");
    try { sessionStorage.removeItem("sandbox_results"); } catch { /* noop */ }
  }, [files, jdText]);

  /* ─── Inference: call backend, then navigate to results ──── */

  const handleInitInference = useCallback(async () => {
    if (!jdText.trim()) {
      setError("Provide a job description.");
      return;
    }
    if (files.length === 0) {
      setError("Upload at least one PDF resume to rank.");
      return;
    }

    setError("");
    setIsProcessing(true);
    setLogLines([]);
    try { sessionStorage.removeItem("sandbox_results"); } catch { /* noop */ }

    try {
      const response = await rankPdfs(jdText, category, files);

      try {
        sessionStorage.setItem("sandbox_results", JSON.stringify(response));
        sessionStorage.setItem("sandbox_jd", jdText);
        sessionStorage.setItem("sandbox_category", category);
      } catch { /* quota */ }

      const elapsed = response.processing_time_seconds.toFixed(2);
      setLogLines([
        "[INFO] TriadRank pipeline complete",
        `[SUCCESS] ${response.returned_candidates}/${response.total_candidates} candidates scored in ${elapsed}s`,
        `[INFO] Job category: ${response.job_category}`,
      ]);
    } catch (e) {
      const msg = e instanceof ApiError ? e.detail : "Could not reach backend. Is the API running on port 8000?";
      setError(msg);
      setIsProcessing(false);
    }
  }, [jdText, files, category]);

  const handleTerminalComplete = useCallback(() => {
    setIsProcessing(false);
    router.push(`/results?category=${encodeURIComponent(category)}`);
  }, [category, router]);

  const handleTerminalClose = useCallback(() => {
    setIsProcessing(false);
    setLogLines([]);
  }, []);

  /* ─── Render ────────────────────────────────────────────────── */

  return (
    <div className="flex flex-col flex-1 bg-[#F9F9F6]">
      <div className="animate-fade-in-up px-8 pt-8 pb-4">
        <h1
          className="font-heading text-[1.5rem] font-semibold"
          style={{ fontFamily: "var(--font-heading)" }}
        >
          Input Sandbox
        </h1>
        <p className="text-[#666] mt-1">
          Upload PDF resumes and run them through the scoring pipeline.
        </p>
      </div>

      <div className="flex-1 px-8 grid max-lg:grid-cols-1 lg:grid-cols-2 lg:grid-rows-[1fr] gap-6">
        {/* ── Left: JD Input ── */}
        <div className="animate-fade-in-up flex flex-col h-full">
          <label className="font-mono text-[12px] uppercase tracking-[0.15em] text-[#666] mb-2">
            Job Description
          </label>
          <textarea
            value={jdText}
            onChange={(e) => setJdText(e.target.value)}
            disabled={isProcessing}
            placeholder="Paste a job description here..."
            className="flex-1 min-h-[400px] w-full resize-none bg-white border border-[#111] p-4 font-mono text-[14px] leading-relaxed outline-none focus:ring-2 focus:ring-[#0000FF] disabled:opacity-50 disabled:cursor-not-allowed"
          />
        </div>

        {/* ── Right: Controls ── */}
        <div className="animate-fade-in-up flex flex-col gap-6 h-full" style={{ animationDelay: "80ms" }}>
          <div>
            <label className="block font-mono text-[12px] uppercase tracking-[0.15em] text-[#666] mb-2">
              Category
            </label>
            {/* Controlled for the component's whole lifetime (value never
                undefined) — flipping uncontrolled→controlled on fetch triggers
                a React/base-ui warning. category defaults to "ENGINEERING",
                a valid backend category, so the disabled-while-loading gate
                is the only loading affordance needed. */}
            <Select value={category} onValueChange={(v) => { if (v) setCategory(v); }}>
              <SelectTrigger className="w-full" disabled={isProcessing || categories.length === 0}>
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                {(categories.length ? categories : ["ENGINEERING"]).map((cat) => (
                  <SelectItem key={cat} value={cat}>{cat}</SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>

          {/* Dropzone */}
          <div
            onDrop={handleDrop}
            onDragOver={handleDragOver}
            onDragLeave={handleDragLeave}
            onClick={() => fileInputRef.current?.click()}
            className={cn(
              "flex flex-1 min-h-[200px] flex-col items-center justify-center border-2 border-dashed border-[#111] cursor-pointer transition-colors",
              isDragging && "bg-[#E5E5FF]"
            )}
          >
            <p className="font-mono text-[14px] text-[#666] px-4 text-center pointer-events-none">
              Drop .PDF files here or click to browse
            </p>
            <input
              ref={fileInputRef}
              type="file"
              accept=".pdf"
              multiple
              onChange={handleFileSelect}
              className="hidden"
            />
          </div>

          {files.length > 0 && (
            <div className="flex flex-wrap gap-2">
              {files.map((file, i) => (
                <div
                  key={`${file.name}-${i}`}
                  className="w-[120px] h-[32px] shrink-0 bg-white border border-[#111] flex items-center px-2 overflow-hidden"
                >
                  <span className="font-mono text-[12px] truncate block w-full">
                    {file.name}
                  </span>
                </div>
              ))}
            </div>
          )}
        </div>
      </div>

      {/* ── Bottom Action Bar ── */}
      <div className="animate-fade-in-up sticky bottom-0 w-full bg-white border-t border-[#111] px-8 py-2 flex items-center justify-between z-10">
        <span className="font-mono text-[12px] text-[#666] flex items-center gap-2">
          {isProcessing ? (
            <>
              <span className="inline-block size-2 bg-[#0000FF] animate-terminal-blink" />
              SYSTEM STATUS: PROCESSING
            </>
          ) : (
            <>
              <span className="inline-block size-2 bg-[#0000FF] animate-terminal-blink" />
              {error ? (
                <span className="text-[#FF0033]">{error}</span>
              ) : (
                <>{files.length} file(s) uploaded &middot; {jdText.length} characters</>
              )}
            </>
          )}
        </span>

        <div className="flex items-center gap-3">
          <Button
            variant="ghost"
            disabled={isProcessing}
            onClick={handleReset}
            className="text-[#FF0033] font-mono text-[12px] font-semibold uppercase tracking-widest hover:text-[#FF0033]"
          >
            Reset
          </Button>

          <Button
            disabled={isProcessing}
            onClick={handleInitInference}
            className="font-mono text-[12px] font-semibold uppercase tracking-widest shadow-[4px_4px_0px_#111] border-[#111] hover:shadow-[2px_2px_0px_#111] hover:translate-x-[2px] hover:translate-y-[2px] transition-all disabled:shadow-[4px_4px_0px_#111] disabled:hover:translate-x-0 disabled:hover:translate-y-0 disabled:hover:shadow-[4px_4px_0px_#111]"
          >
            {isProcessing ? (
              <>
                <span className="inline-block w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin" />
                PROCESSING...
              </>
            ) : (
              "Initialize Inference"
            )}
          </Button>
        </div>
      </div>

      {/* ── Terminal — shown only after API succeeds, before navigation ── */}
      <Terminal
        lines={logLines}
        isVisible={logLines.length > 0}
        onClose={handleTerminalClose}
        onComplete={handleTerminalComplete}
      />
    </div>
  );
}
