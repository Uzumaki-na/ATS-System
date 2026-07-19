"use client";

import { useState, useEffect } from "react";
import Link from "next/link";
import { usePathname } from "next/navigation";
import { Settings, Terminal, ArrowRight, Menu, X } from "lucide-react";

const NAV_LINKS = [
  { href: "/", label: "Abstract" },
  { href: "/sandbox", label: "Sandbox" },
  { href: "/results", label: "Results" },
  { href: "/methodology", label: "Methodology" },
] as const;

export function NavBar() {
  const pathname = usePathname();
  const [mobileOpen, setMobileOpen] = useState(false);

  useEffect(() => {
    setMobileOpen(false);
  }, [pathname]);

  const isActive = (href: string) =>
    href === "/" ? pathname === "/" : pathname.startsWith(href);

  return (
    <header className="sticky top-0 z-50 h-14 border-b border-[#111] bg-white/95 backdrop-blur-md shadow-[4px_4px_0px_0px_rgba(17,17,17,1)]">
      <div className="mx-auto flex h-full max-w-7xl items-center gap-8 px-6">
        {/* Brand */}
        <Link
          href="/"
          className="flex items-center gap-2 font-mono text-sm font-semibold uppercase tracking-[0.05em] text-[#111] no-underline"
        >
          <span className="inline-block size-2 bg-[#0000FF]" />
          THE LAB
        </Link>

        {/* Desktop Nav */}
        <nav className="hidden md:flex md:items-center md:gap-6">
          {NAV_LINKS.map((link) => (
            <Link
              key={link.href}
              href={link.href}
              className={`font-mono text-xs font-medium uppercase tracking-[0.075em] transition-colors duration-150 py-[3px] border-b ${
                isActive(link.href)
                  ? "text-[#111] font-semibold border-[#111]"
                  : "text-[#666] border-transparent hover:text-[#111] hover:border-[#111]"
              }`}
            >
              {link.label}
            </Link>
          ))}
        </nav>

        {/* Toolbar */}
        <div className="ml-auto flex items-center gap-2">
          {/* Settings */}
          <button
            aria-label="Settings"
            title="Settings"
            className="flex size-9 items-center justify-center border border-transparent bg-transparent text-[#111] transition-all duration-150 cursor-pointer hover:border-[#111] hover:bg-[#dad8e8] hover:shadow-[2px_2px_0px_0px_#111]"
          >
            <Settings className="size-4" />
          </button>

          {/* Terminal */}
          <button
            aria-label="Terminal"
            title="Terminal"
            className="flex size-9 items-center justify-center border border-transparent bg-transparent text-[#111] transition-all duration-150 cursor-pointer hover:border-[#111] hover:bg-[#dad8e8] hover:shadow-[2px_2px_0px_0px_#111]"
          >
            <Terminal className="size-4" />
          </button>

          {/* CTA */}
          <Link
            href="/sandbox"
            className="hidden sm:inline-flex items-center gap-2 font-mono text-xs font-semibold uppercase tracking-[0.075em] bg-[#111] text-white px-4 py-2 border border-[#111] shadow-[4px_4px_0px_0px_#0000FF] transition-all duration-150 hover:-translate-x-0.5 hover:-translate-y-0.5 hover:shadow-[6px_6px_0px_0px_#0000FF] active:translate-x-0 active:translate-y-0 active:shadow-[4px_4px_0px_0px_#0000FF] cursor-pointer"
          >
            Execute Analysis
            <ArrowRight className="size-3.5" />
          </Link>

          {/* Mobile hamburger */}
          <button
            aria-label={mobileOpen ? "Close menu" : "Open menu"}
            className="flex md:hidden size-9 items-center justify-center cursor-pointer text-[#111]"
            onClick={() => setMobileOpen(!mobileOpen)}
          >
            {mobileOpen ? <X className="size-5" /> : <Menu className="size-5" />}
          </button>
        </div>
      </div>

      {/* Mobile Nav */}
      {mobileOpen && (
        <nav className="border-t border-[#111] bg-white md:hidden">
          <div className="flex flex-col px-6 pb-4 pt-2">
            {NAV_LINKS.map((link) => (
              <Link
                key={link.href}
                href={link.href}
                onClick={() => setMobileOpen(false)}
                className={`font-mono text-sm font-medium uppercase tracking-[0.075em] py-3 transition-colors ${
                  isActive(link.href)
                    ? "text-[#0000FF]"
                    : "text-[#111] hover:text-[#0000FF]"
                }`}
              >
                {link.label}
              </Link>
            ))}
            <hr className="my-3 border-[#111]/20" />
            <div className="flex gap-2 pt-1">
              <button
                aria-label="Settings"
                className="flex items-center gap-2 font-mono text-sm text-[#111] px-3 py-2 border border-transparent hover:border-[#111] transition-colors cursor-pointer"
              >
                <Settings className="size-4" />
                Settings
              </button>
              <button
                aria-label="Terminal"
                className="flex items-center gap-2 font-mono text-sm text-[#111] px-3 py-2 border border-transparent hover:border-[#111] transition-colors cursor-pointer"
              >
                <Terminal className="size-4" />
                Terminal
              </button>
            </div>
          </div>
        </nav>
      )}
    </header>
  );
}
