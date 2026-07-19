import type { NextConfig } from "next";

// Server-side only: the real backend URL never reaches the browser.
//   Dev on host:  unset  → http://localhost:8000 (Docker-published port)
//   In compose:   BACKEND_URL=http://triadrank-ats:8000 (service DNS)
const BACKEND_URL = process.env.BACKEND_URL || "http://localhost:8000";

const nextConfig: NextConfig = {
  turbopack: {
    root: process.cwd(),
  },
  // Proxy same-origin /api/* to the backend so the browser never needs CORS
  // and never sees the backend's real origin.
  async rewrites() {
    return [
      { source: "/api/:path*", destination: `${BACKEND_URL}/:path*` },
    ];
  },
};

export default nextConfig;
