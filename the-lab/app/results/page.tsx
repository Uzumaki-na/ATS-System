"use client";

import Link from "next/link"
import { useState, useMemo, useEffect, Suspense } from "react";
import { useSearchParams } from "next/navigation";
import { apiCandidateToView, type Candidate } from "@/lib/mock-data";
import type { ApiRankResponse } from "@/lib/api";
import { Badge } from "@/components/ui/badge";
import { Skeleton } from "@/components/ui/skeleton";
import DetailModal from "@/components/detail-modal";

type SortKey = "score" | "name" | "category" | "skillOverlap";
type SortDir = "asc" | "desc";

const labelColors: Record<string, string> = {
  "Good Fit": "bg-[#00AA00] text-white",
  "Potential Fit": "bg-[#FF8800] text-white",
  "Bad Fit": "bg-[#FF0033] text-white",
};

function ResultsInner() {
  const searchParams = useSearchParams();
  const category = searchParams.get("category") || "ENGINEERING";

  const [jd, setJd] = useState("");
  const [candidates, setCandidates] = useState<Candidate[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");

  useEffect(() => {
    try {
      // Read API results from sessionStorage
      const stored = sessionStorage.getItem("sandbox_results");
      const storedJd = sessionStorage.getItem("sandbox_jd");

      if (storedJd) setJd(storedJd);

      if (stored) {
        const parsed = JSON.parse(stored) as ApiRankResponse;
        if (parsed.results?.length > 0) {
          setCandidates(parsed.results.map(apiCandidateToView));
        } else {
          setError("The pipeline returned no candidates.");
        }
      } else {
        setError("No ranking data found. Upload resumes from the sandbox first.");
      }
    } catch {
      setError("Failed to read ranking results.");
    } finally {
      setLoading(false);
    }
  }, []);

  const [jdExpanded, setJdExpanded] = useState(false);

  const [sortKey, setSortKey] = useState<SortKey>("score");
  const [sortDir, setSortDir] = useState<SortDir>("desc");
  const [selected, setSelected] = useState<Candidate | null>(null);
  const [modalOpen, setModalOpen] = useState(false);

  const sorted = useMemo(() => {
    const copy = [...candidates];
    copy.sort((a, b) => {
      const mul = sortDir === "asc" ? 1 : -1;
      if (sortKey === "name") return mul * a.name.localeCompare(b.name);
      if (sortKey === "category") return mul * a.category.localeCompare(b.category);
      if (sortKey === "skillOverlap") return mul * ((a.skillOverlap ?? 0) - (b.skillOverlap ?? 0));
      return mul * (a.score - b.score);
    });
    return copy;
  }, [candidates, sortKey, sortDir]);

  const toggleSort = (key: SortKey) => {
    if (sortKey === key) setSortDir((d) => (d === "asc" ? "desc" : "asc"));
    else { setSortKey(key); setSortDir("desc"); }
  };

  const sortArrow = (key: SortKey) =>
    sortKey === key ? (sortDir === "asc" ? " ↑" : " ↓") : "";

  const openDetail = (c: Candidate) => {
    setSelected(c);
    setModalOpen(true);
  };

  /* ─── Loading state ── */
  if (loading) {
    return (
      <div className="bg-[#F9F9F6] min-h-full flex flex-col">
        <div className="animate-fade-in-up border-b border-[#111] bg-white px-8 py-4">
          <Skeleton className="h-8 w-48" />
          <Skeleton className="h-4 w-64 mt-2" />
        </div>
        <div className="p-8">
          <div className="border border-[#111] bg-white">
            <div className="border-b border-[#111] bg-[#f0f0ee] px-4 py-3">
              <Skeleton className="h-4 w-48" />
            </div>
            {Array.from({ length: 6 }).map((_, i) => (
              <div
                key={i}
                className="flex items-center gap-4 border-b border-[#eee] px-4 py-4 animate-skeleton"
                style={{ animationDelay: `${i * 60}ms` }}
              >
                <Skeleton className="h-4 w-32" />
                <Skeleton className="h-4 w-16" />
                <Skeleton className="h-4 w-20" />
                <Skeleton className="h-4 w-28" />
                <Skeleton className="h-4 w-12" />
                <Skeleton className="h-4 w-12" />
                <Skeleton className="h-4 w-8" />
                <Skeleton className="h-4 w-12" />
              </div>
            ))}
          </div>
        </div>
      </div>
    );
  }

  /* ─── Error / empty state ── */
  if (error || candidates.length === 0) {
    return (
      <div className="bg-[#F9F9F6] min-h-full flex flex-col">
        <div className="animate-fade-in-up border-b border-[#111] bg-white px-8 py-4">
          <h1 className="font-heading text-[1.5rem] font-semibold">Results</h1>
        </div>
        <div className="px-8 pb-8 flex-1">
          <div className="relative mt-6 overflow-hidden border border-[#111] bg-white animate-fade-in-up">
            <div
              className="pointer-events-none absolute inset-0 opacity-[0.03]"
              style={{
                backgroundImage:
                  "linear-gradient(rgba(0,0,0,.1) 1px, transparent 1px), linear-gradient(90deg, rgba(0,0,0,.1) 1px, transparent 1px)",
                backgroundSize: "24px 24px",
              }}
            />
            <div className="relative px-8 py-16 text-center">
              <div className="mx-auto mb-6 flex max-w-[360px] items-center gap-2">
                {["Extractor", "Category", "Cross-Enc"].map((name, i) => (
                  <div key={name} className="flex items-center gap-2 flex-1">
                    <div className="flex-1 border border-[#ddd] bg-[#f9f9f6] px-2 py-2">
                      <div className="font-mono text-[9px] text-[#bbb] uppercase tracking-wider">{name}</div>
                      <div className="mt-0.5 h-1 w-full bg-[#eee]" />
                    </div>
                    {i < 2 && <span className="text-[#ddd] text-xs">→</span>}
                  </div>
                ))}
              </div>

              <div className="mx-auto mb-4 flex h-14 w-14 items-center justify-center border-2 border-[#111] bg-[#f0f0ee] shadow-[3px_3px_0px_#111]">
                <span className="text-xl font-mono font-bold text-[#999]">0</span>
              </div>

              <h3 className="font-heading text-lg font-semibold text-[#111]">
                {error || "No candidates"}
              </h3>
              <p className="mx-auto mt-2 max-w-sm font-mono text-[13px] leading-relaxed text-muted-foreground">
                Upload PDF resumes from the sandbox and run the pipeline to see ranked results here.
              </p>
              <div className="mt-6 flex flex-wrap items-center justify-center gap-3">
                <Link href="/sandbox">
                  <button className="h-9 cursor-pointer border border-[#111] bg-[#111] px-4 font-mono text-[12px] text-white transition-all duration-200 hover:-translate-x-0.5 hover:-translate-y-0.5 hover:shadow-[3px_3px_0px_#111] active:translate-x-0 active:translate-y-0">
                    Go to Sandbox
                  </button>
                </Link>
              </div>
            </div>
          </div>
        </div>
      </div>
    );
  }

  /* ─── Normal results view ── */
  return (
    <div className="bg-[#F9F9F6] min-h-full flex flex-col">
      <div className="animate-fade-in-up border-b border-[#111] bg-white px-8 py-4">
        <h1 className="font-heading text-[1.5rem] font-semibold">Results</h1>
        <p className="text-sm text-muted-foreground mt-1 font-mono">
          {category} &middot; {candidates.length} candidates
        </p>
      </div>

      {jd ? (
        <div>
          <div
            className={`mx-8 mt-6 overflow-y-auto rounded border border-[#111] bg-white p-4 font-mono text-[13px] leading-relaxed text-muted-foreground animate-fade-in-up ${jdExpanded ? "max-h-none" : "max-h-28"}`}
          >
            {jdExpanded ? jd : `${jd.slice(0, 600)}${jd.length > 600 ? " ..." : ""}`}
          </div>
          {jd.length > 600 && (
            <button
              onClick={() => setJdExpanded((v) => !v)}
              className="mx-8 mt-1.5 font-mono text-[12px] text-[#0000FF] hover:underline cursor-pointer"
            >
              {jdExpanded ? "Show less" : "Show all"}
            </button>
          )}
        </div>
      ) : null}

      <div className="animate-fade-in-up px-8 pb-8 flex-1" style={{ animationDelay: "80ms" }}>
        <div className="mt-6 overflow-x-auto border border-[#111] bg-white">
          <table className="w-full text-left font-mono text-[13px]">
            <thead>
              <tr className="border-b border-[#111] bg-[#f0f0ee]">
                <th onClick={() => toggleSort("name")} className={"px-4 py-3 text-[12px] uppercase tracking-wider text-muted-foreground cursor-pointer select-none hover:text-[#111]"}>Name{sortArrow("name")}</th>
                <th onClick={() => toggleSort("score")} className={"px-4 py-3 text-[12px] uppercase tracking-wider text-muted-foreground cursor-pointer select-none hover:text-[#111]"}>Score{sortArrow("score")}</th>
                <th className="px-4 py-3 text-[12px] uppercase tracking-wider text-muted-foreground">Label</th>
                <th onClick={() => toggleSort("category")} className={"px-4 py-3 text-[12px] uppercase tracking-wider text-muted-foreground cursor-pointer select-none hover:text-[#111]"}>Category{sortArrow("category")}</th>
                <th className="px-4 py-3 text-[12px] uppercase tracking-wider text-muted-foreground">Match</th>
                <th onClick={() => toggleSort("skillOverlap")} className={"px-4 py-3 text-[12px] uppercase tracking-wider text-muted-foreground cursor-pointer select-none hover:text-[#111]"}>Skills{sortArrow("skillOverlap")}</th>
                <th className="px-4 py-3 text-[12px] uppercase tracking-wider text-muted-foreground">Exp.</th>
                <th className="px-4 py-3 text-[12px] uppercase tracking-wider text-muted-foreground">Time</th>
              </tr>
            </thead>
            <tbody className="row-stagger">
              {sorted.map((c) => (
                <tr key={c.id} onClick={() => openDetail(c)} className="animate-row-reveal border-b border-[#eee] cursor-pointer transition-all duration-200 hover:bg-[#E5E5FF] hover:-translate-y-[1px] hover:shadow-[0_2px_0px_#111]">
                  <td className="px-4 py-3 font-semibold">{c.name}</td>
                  <td className="px-4 py-3"><span className="tabular-nums text-[#0000FF]">{c.score.toFixed(4)}</span></td>
                  <td className="px-4 py-3"><Badge className={labelColors[c.label] ?? "bg-[#999] text-white"}>{c.label}</Badge></td>
                  <td className="px-4 py-3"><span className="text-muted-foreground">{c.category}</span></td>
                  <td className="px-4 py-3">{c.categoryMatch ? <span className="text-[#00AA00]">&#10003;</span> : <span className="text-[#FF0033]">&#10007;</span>}</td>
                  <td className="px-4 py-3"><span className="tabular-nums">{c.skillOverlap}%</span></td>
                  <td className="px-4 py-3 tabular-nums">{c.experienceYears}y</td>
                  <td className="px-4 py-3 tabular-nums text-muted-foreground">{c.processingTimeMs}ms</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <p className="mt-4 text-center font-mono text-[12px] text-muted-foreground">Click any row to view the full scoring breakdown.</p>
      </div>

      <DetailModal candidate={selected} open={modalOpen} onOpenChange={setModalOpen} />
    </div>
  );
}

export default function ResultsPage() {
  return (
    <Suspense fallback={<div className="bg-[#F9F9F6] p-8 space-y-4"><Skeleton className="h-8 w-48" /><Skeleton className="h-4 w-64" /><Skeleton className="mt-6 h-96 w-full" /></div>}>
      <ResultsInner />
    </Suspense>
  );
}
