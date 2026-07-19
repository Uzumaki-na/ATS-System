"use client";

import { Dialog, DialogContent, DialogTitle } from "@/components/ui/dialog";
import { X } from "lucide-react";
import { Button } from "@/components/ui/button";
import type { Candidate } from "@/lib/mock-data";
import ScoreRadar from "@/components/score-radar";

interface DetailModalProps {
  candidate: Candidate | null;
  open: boolean;
  onOpenChange: (open: boolean) => void;
}

function LabelBadge({ label }: { label: string }) {
  const colors: Record<string, string> = {
    "Good Fit": "bg-[#00AA00] text-white",
    "Potential Fit": "bg-[#FF8800] text-white",
    "Bad Fit": "bg-[#FF0033] text-white",
  };

  return (
    <span
      className={`inline-block px-3 py-0.5 text-xs font-mono font-semibold ${colors[label] ?? "bg-[#999] text-white"}`}
    >
      {label}
    </span>
  );
}

function ScoreBar({
  value: rawValue,
  color,
  index,
}: {
  value: number;
  color: string;
  index: number;
}) {
  const value = Math.min(1, Math.max(0, rawValue));
  return (
    <div className="w-full h-4 bg-[#f0f0f0] border border-[#111] relative overflow-hidden">
      <div
        className="absolute inset-y-0 left-0 animate-modal-bar"
        style={{
          width: `${Math.round(value * 100)}%`,
          backgroundColor: color,
          animationDelay: `${0.15 + index * 0.12}s`,
        }}
      />
    </div>
  );
}

export default function DetailModal({
  candidate,
  open,
  onOpenChange,
}: DetailModalProps) {
  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      {!candidate ? (
        <DialogContent
          className="max-w-[1600px] w-[calc(100vw-48px)] sm:max-w-[calc(100vw-48px)] p-0 gap-0 border border-[#111] shadow-[8px_8px_0px_#111] max-h-[90dvh] overflow-y-auto"
          style={{ borderRadius: 0 }}
          showCloseButton={false}
        >
          <div className="flex items-center justify-between border-b border-[#111] px-6 py-3">
            <DialogTitle className="font-mono text-xs uppercase tracking-widest text-[#666]">
              Candidate Detail
            </DialogTitle>
            <Button
              variant="ghost"
              size="icon-sm"
              onClick={() => onOpenChange(false)}
              className="text-[#111] hover:text-[#FF0033]"
            >
              <X className="size-4" />
            </Button>
          </div>
          <div className="p-6 font-mono text-[13px] text-[#666]">No candidate data.</div>
        </DialogContent>
      ) : (
        <DialogContent
          className="max-w-[1600px] w-[calc(100vw-48px)] sm:max-w-[calc(100vw-48px)] p-0 gap-0 border border-[#111] shadow-[8px_8px_0px_#111] animate-scale-in max-h-[90dvh] overflow-y-auto"
          style={{ borderRadius: 0 }}
          showCloseButton={false}
        >
        {/* Custom header with close button */}
        <div className="flex items-center justify-between border-b border-[#111] px-6 py-3">
          <DialogTitle className="font-mono text-xs uppercase tracking-widest text-[#666]">
            Candidate Detail
          </DialogTitle>
          <Button
            variant="ghost"
            size="icon-sm"
            onClick={() => onOpenChange(false)}
            className="text-[#111] hover:text-[#FF0033]"
          >
            <X className="size-4" />
          </Button>
        </div>

        {/* Body: two columns */}
        <div className="flex flex-col md:flex-row">
          {/* ── Left Column (60%) ── */}
          <div className="w-full md:w-[60%] border-r border-[#111] p-6 space-y-6">
            {/* Score Summary */}
            <div>
              <h2
                className="text-[1.5rem] font-semibold leading-tight"
                style={{ fontFamily: "var(--font-crimson-pro), Crimson Pro, serif" }}
              >
                {candidate.name}
              </h2>
              <div className="flex items-end gap-4 mt-1">
                <span
                  className="text-[2rem] leading-none font-mono font-semibold"
                  style={{ color: "#0000FF" }}
                >
                  {candidate.score.toFixed(4)}
                </span>
                <LabelBadge label={candidate.label} />
              </div>
            </div>

            {/* Radar Chart */}
            <div className="flex flex-col items-center">
              <ScoreRadar
                semantic={candidate.semantic}
                syntax={candidate.syntax}
                experience={candidate.experience}
                education={candidate.education}
                keywords={candidate.keywords}
              />
              <div className="flex gap-6 mt-3 font-mono text-[12px] text-[#111]">
                <span>Skill Overlap <span className="text-[#0000FF]">{candidate.skillOverlap}%</span></span>
                <span>Keyword Overlap <span className="text-[#0000FF]">{candidate.keywordOverlap}%</span></span>
              </div>
            </div>

            {/* Entity Extraction Chain */}
            <div>
              <h3 className="font-mono text-[12px] uppercase tracking-wider text-[#666] mb-3">
                Entity Extraction
              </h3>
              <div className="space-y-3">
                {candidate.entityExtractionChain.map((item, i) => (
                  <div key={i} className="font-mono text-[12px] leading-relaxed break-words max-w-full">
                    <span className="text-[#666]">JD: </span>
                    <span>{item.jd}</span>
                    <span className="mx-2 text-[#0000FF]">→</span>
                    <span className="text-[#666]">Resume: </span>
                    <span>{item.resume}</span>
                  </div>
                ))}
              </div>
            </div>
          </div>

          {/* ── Right Column (40%) ── */}
          <div className="w-full md:w-[40%] p-6 space-y-6">
            {/* Extracted Skills */}
            <div>
              <h3 className="font-mono text-[12px] uppercase tracking-wider text-[#666] mb-2">
                Extracted Skills ({candidate.skills.length})
              </h3>
              <div className="flex flex-wrap gap-1.5">
                {candidate.skills.map((skill) => (
                  <span
                    key={skill}
                    className="inline-block px-2 py-0.5 font-mono text-[11px] text-white bg-[#0000FF]"
                  >
                    {skill}
                  </span>
                ))}
              </div>
            </div>

            {/* Category Validation */}
            <div>
              <h3 className="font-mono text-[12px] uppercase tracking-wider text-[#666] mb-2">
                Category Validation
              </h3>
              <div className="font-mono text-[13px] space-y-1">
                <div>
                  <span className="text-[#666]">Predicted: </span>
                  <span>{candidate.category}</span>
                </div>
                <div>
                  {candidate.categoryMatch ? (
                    <span className="text-[#00AA00]">✓ Matched</span>
                  ) : (
                    <span className="text-[#FF0033]">✗ Mismatched</span>
                  )}
                </div>
                <div>
                  <span className="text-[#666]">Confidence: </span>
                  <span>{(candidate.categoryConfidence * 100).toFixed(0)}%</span>
                </div>
              </div>
            </div>

            {/* Penalty Log */}
            <div>
              <h3 className="font-mono text-[12px] uppercase tracking-wider text-[#666] mb-2">
                Penalty Log
              </h3>
              {candidate.penalties.length === 0 ? (
                <span className="font-mono text-[12px] text-[#00AA00]">
                  No penalties applied
                </span>
              ) : (
                <div className="space-y-1">
                  {candidate.penalties.map((p, i) => (
                    <div key={i} className="flex justify-between font-mono text-[12px]">
                      <span className="text-[#666]">{p.reason}</span>
                      <span className={p.multiplier < 0.7 ? "text-[#FF0033]" : "text-[#FF8800]"}>
                        x{p.multiplier.toFixed(2)}
                      </span>
                    </div>
                  ))}
                </div>
              )}
            </div>

            {/* Label Probabilities */}
            <div>
              <h3 className="font-mono text-[12px] uppercase tracking-wider text-[#666] mb-2">
                Label Probabilities
              </h3>
              <div className="space-y-2">
                <div>
                  <div className="flex justify-between font-mono text-[12px] mb-0.5">
                    <span>Good Fit</span>
                    <span>{(candidate.labelProbabilities.goodFit * 100).toFixed(1)}%</span>
                  </div>
                  <ScoreBar value={candidate.labelProbabilities.goodFit} color="#0000FF" index={0} />
                </div>
                <div>
                  <div className="flex justify-between font-mono text-[12px] mb-0.5">
                    <span>Potential Fit</span>
                    <span>{(candidate.labelProbabilities.potentialFit * 100).toFixed(1)}%</span>
                  </div>
                  <ScoreBar value={candidate.labelProbabilities.potentialFit} color="#FF8800" index={1} />
                </div>
                <div>
                  <div className="flex justify-between font-mono text-[12px] mb-0.5">
                    <span>Bad Fit</span>
                    <span>{(candidate.labelProbabilities.badFit * 100).toFixed(1)}%</span>
                  </div>
                  <ScoreBar value={candidate.labelProbabilities.badFit} color="#FF0033" index={2} />
                </div>
              </div>
            </div>
          </div>
        </div>
      </DialogContent>
    )}
    </Dialog>
  );
}
