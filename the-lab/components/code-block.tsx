'use client'

import { useState, useCallback } from 'react'

interface CodeBlockProps {
  code: string
  language?: string
  filename?: string
}

export default function CodeBlock({ code, filename }: CodeBlockProps) {
  const [copied, setCopied] = useState(false)

  const handleCopy = useCallback(async () => {
    try {
      await navigator.clipboard.writeText(code);
      setCopied(true);
    } catch {
      // clipboard not available (insecure context etc.)
    }
    setTimeout(() => setCopied(false), 2000);
  }, [code]);

  return (
    <div className="group relative my-6 overflow-hidden" style={{ background: '#111111' }}>
      {/* Header bar */}
      <div
        className="flex items-center justify-between px-4 py-2"
        style={{ background: '#1A1A1A', borderBottom: '1px solid #333' }}
      >
        <div className="flex items-center gap-2">
          <span className="h-3 w-3 rounded-full" style={{ background: '#FF5F56' }} />
          <span className="h-3 w-3 rounded-full" style={{ background: '#FFBD2E' }} />
          <span className="h-3 w-3 rounded-full" style={{ background: '#27C93F' }} />
          {filename && (
            <span
              className="ml-2 text-xs"
              style={{ color: '#888', fontFamily: 'var(--font-mono)', letterSpacing: '0.025em' }}
            >
              {filename}
            </span>
          )}
        </div>
        <button
          onClick={handleCopy}
          className="cursor-pointer px-2 py-0.5 text-xs opacity-0 transition-opacity duration-150 group-hover:opacity-100"
          style={{
            color: '#ccc',
            fontFamily: 'var(--font-mono)',
            background: '#333',
            border: 0,
            borderRadius: 0,
          }}
        >
          {copied ? 'Copied' : 'Copy'}
        </button>
      </div>
      {/* Code */}
      <pre className="m-0 overflow-x-auto p-4">
        <code
          style={{
            color: '#FAFAFA',
            fontFamily: 'var(--font-mono)',
            fontSize: '14px',
            lineHeight: 1.6,
            whiteSpace: 'pre',
          }}
        >
          {code}
        </code>
      </pre>
    </div>
  )
}
