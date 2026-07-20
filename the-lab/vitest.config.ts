import { defineConfig } from "vitest/config";

// Pure-logic gate: node env (no DOM needed — view-models.ts is a pure-math
// mapper; api.ts uses Node 18+ globals fetch/FormData/Response). Scoped to
// tests/unit/** so vitest never picks up the Playwright e2e specs in tests/e2e.
export default defineConfig({
  test: {
    environment: "node",
    include: ["tests/unit/**/*.test.ts"],
  },
});
