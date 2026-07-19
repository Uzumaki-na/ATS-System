"use client";

import { useState, useEffect, useRef, useCallback } from "react";
import { X } from "lucide-react";

interface TerminalProps {
  lines: string[];
  isVisible: boolean;
  onClose: () => void;
  onComplete?: () => void;
}

const COLORS = {
  info: "#88ccff",
  success: "#00ff88",
  warn: "#ffcc44",
  error: "#ff4466",
} as const;

function parseLine(line: string) {
  const m = line.match(/^\[(INFO|SUCCESS|ERROR|WARN)\]\s*(.*)/);
  const tag = m?.[1]?.toLowerCase() as keyof typeof COLORS | undefined ?? null;
  const text = m ? `[${m[1]}] ${m[2]}` : line;
  return { type: tag, text };
}

export default function Terminal({ lines, isVisible, onClose, onComplete }: TerminalProps) {
  const [visibleCount, setVisibleCount] = useState(0);
  const scrollRef = useRef<HTMLDivElement>(null);
  const timerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const aliveRef = useRef(false);

  // Animate lines in one at a time
  useEffect(() => {
    if (!isVisible || lines.length === 0) return;

    setVisibleCount(0);
    aliveRef.current = true;

    let i = 0;
    const tick = () => {
      if (!aliveRef.current) return;
      i++;
      setVisibleCount(i);
      if (i >= lines.length) {
        // All lines shown — wait, then fire complete
        timerRef.current = setTimeout(() => onComplete?.(), 500);
        return;
      }
      timerRef.current = setTimeout(tick, 220);
    };
    timerRef.current = setTimeout(tick, 120);

    return () => {
      aliveRef.current = false;
      if (timerRef.current) clearTimeout(timerRef.current);
    };
  }, [isVisible, lines]);

  // Auto-scroll to bottom as lines appear (only if user hasn't scrolled up)
  const userScrolledUpRef = useRef(false);

  useEffect(() => {
    if (!userScrolledUpRef.current) {
      scrollRef.current?.scrollTo({ top: scrollRef.current.scrollHeight, behavior: "smooth" });
    }
  }, [visibleCount]);

  const handleScroll = useCallback(() => {
    const el = scrollRef.current;
    if (!el) return;
    const threshold = 30;
    userScrolledUpRef.current = el.scrollHeight - el.scrollTop - el.clientHeight > threshold;
  }, []);

  const handleClose = () => {
    aliveRef.current = false;
    if (timerRef.current) clearTimeout(timerRef.current);
    onClose();
  };

  if (!isVisible) return null;

  const shownLines = lines.slice(0, visibleCount);

  return (
    <div className="fixed bottom-0 left-0 right-0 z-50 flex flex-col h-[300px] bg-[#111] border-t-2 border-[#111] shadow-[0_-4px_0px_0px_rgba(17,17,17,1)]">
      {/* macOS-style terminal header */}
      <div className="flex items-center justify-between px-4 py-[10px] border-b border-[#333] shrink-0">
        <div className="flex items-center gap-3">
          <div className="flex gap-[5px]">
            <span className="w-[10px] h-[10px] rounded-full bg-[#ff5f57]" />
            <span className="w-[10px] h-[10px] rounded-full bg-[#ffbd2e]" />
            <span className="w-[10px] h-[10px] rounded-full bg-[#28c840]" />
          </div>
          <span className="font-mono text-[11px] text-[#666]">pipeline/triadrank.py</span>
        </div>
        <button
          onClick={handleClose}
          className="font-mono text-[11px] text-[#555] hover:text-white transition-colors"
          aria-label="Close terminal"
        >
          <X size={14} />
        </button>
      </div>

      {/* Log area */}
      <div ref={scrollRef} onScroll={handleScroll} className="flex-1 overflow-y-auto p-5 font-mono text-[13px] leading-[1.7] whitespace-pre-wrap">
        {shownLines.map((line, i) => {
          const { type, text } = parseLine(line);
          const isLast = i === shownLines.length - 1;
          return (
            <div
              key={i}
              className="animate-terminal-fade-in"
              style={{ color: type ? COLORS[type] : "#888" }}
            >
              {text}
              {/* Blinking cursor on the most recent line */}
              {isLast && (
                <span className="inline-block w-[6px] h-[14px] ml-[2px] align-middle bg-current animate-terminal-blink" />
              )}
            </div>
          );
        })}
        {!isVisible && null}
      </div>
    </div>
  );
}
