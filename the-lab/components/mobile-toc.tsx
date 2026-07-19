"use client";

import { useState } from "react";

const tocLinks = [
  { href: "#overview", label: "1. Overview" },
  { href: "#architecture", label: "2. Architecture" },
  { href: "#tier3", label: "3. Tier 3: Extractor" },
  { href: "#tier2", label: "4. Tier 2: Category Encoder" },
  { href: "#tier1", label: "5. Tier 1: Cross-Encoder" },
  { href: "#scoring", label: "6. Scoring & Ranking" },
  { href: "#bias", label: "7. Bias Mitigation" },
  { href: "#references", label: "8. References" },
];

export default function MobileToc() {
  const [open, setOpen] = useState(false);

  return (
    <div className="lg:hidden mb-8 animate-fade-in-up">
      <button
        onClick={() => setOpen(!open)}
        className="flex w-full items-center justify-between border border-[#111] bg-white px-4 py-3 font-mono text-[13px] text-[#111]"
      >
        <span>Contents</span>
        <span className={`transition-transform duration-200 ${open ? "rotate-180" : ""}`}>
          ▼
        </span>
      </button>
      {open && (
        <ul className="border-x border-b border-[#111] bg-white px-4 py-3 space-y-3">
          {tocLinks.map((link) => (
            <li key={link.href}>
              <a
                href={link.href}
                onClick={() => setOpen(false)}
                className="font-mono text-[13px] text-[#111] hover:text-blue-600 hover:underline block"
              >
                {link.label}
              </a>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}
