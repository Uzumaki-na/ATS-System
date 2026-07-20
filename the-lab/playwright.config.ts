import { defineConfig, devices } from "@playwright/test";

/**
 * End-to-end integration + OUTPUT-CORRECTNESS gate.
 *
 * Lanes are deliberately separate (see package.json):
 *   - `npm test`         : Vitest unit gate — pure logic, mocked fetch, ~220ms,
 *                          no DOM/backend. Cheap + flicker-free; runs anywhere.
 *   - `npm run test:e2e` : Playwright browser gate (this dir). Drives the real
 *                          flow browser → Next dev → `rewrites` proxy → Docker
 *                          backend → trained checkpoints, on one fixture, and
 *                          asserts the DISPLAYED outputs are logically CORRECT,
 *                          not merely that they rendered. Multi-second, needs
 *                          the whole stack up; runs on demand.
 *
 * Preconditions (the suite skips cleanly if unmet, instead of masquerading a
 * down-stack as a code failure):
 *   - Docker backend healthy on :8000. next.config.ts `rewrites` proxies
 *     same-origin /api/* → http://localhost:8000/* (server-side; backend URL
 *     never ships to the browser).
 *   - `npm run dev` is auto-started by `webServer` below.
 */
export default defineConfig({
  testDir: "./tests/e2e",
  workers: 1, // single-candidate backend; avoid concurrent inference / dev-server races
  fullyParallel: false,
  retries: process.env.CI ? 2 : 0, // locally: surface real breaks (don't retry over them)
  forbidOnly: !!process.env.CI,
  reporter: "list",
  use: {
    baseURL: "http://localhost:3000",
    trace: "on-first-retry",
    actionTimeout: 15_000,
    navigationTimeout: 60_000,
  },
  projects: [{ name: "chromium", use: { ...devices["Desktop Chrome"] } }],
  webServer: {
    command: "npm run dev",
    url: "http://localhost:3000",
    reuseExistingServer: !process.env.CI,
    timeout: 120_000, // Next 16 first-boot compile can be slow
    stdout: "pipe",
    stderr: "pipe",
  },
});
