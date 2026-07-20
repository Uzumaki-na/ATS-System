import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { healthCheck, fetchCategories, rankPdfs, ApiError } from "../../lib/api";

/* All three fns hit global fetch at the same-origin /api/* (proxied to the
   backend by next.config.ts rewrites). Node 24 has fetch/FormData/Response/
   File as globals, so no polyfill — we stub globalThis.fetch per-test. */

function jsonRes(body: unknown, init: { status?: number } = {}): Response {
  return new Response(JSON.stringify(body), {
    status: init.status ?? 200,
    headers: { "content-type": "application/json" },
  });
}

describe("healthCheck — returns boolean, never throws", () => {
  beforeEach(() => vi.stubGlobal("fetch", vi.fn()));
  afterEach(() => vi.unstubAllGlobals());

  it("true on 200", async () => {
    (globalThis.fetch as any).mockResolvedValueOnce(jsonRes({ status: "ok" }, { status: 200 }));
    expect(await healthCheck()).toBe(true);
  });
  it("false on non-2xx (swallowed, not thrown)", async () => {
    (globalThis.fetch as any).mockResolvedValueOnce(jsonRes({}, { status: 503 }));
    expect(await healthCheck()).toBe(false);
  });
  it("false when fetch rejects (network/abort)", async () => {
    (globalThis.fetch as any).mockRejectedValueOnce(new Error("network"));
    expect(await healthCheck()).toBe(false);
  });
});

describe("fetchCategories", () => {
  beforeEach(() => vi.stubGlobal("fetch", vi.fn()));
  afterEach(() => vi.unstubAllGlobals());

  it("returns the categories array from the backend", async () => {
    const cats = ["ACCOUNTANT", "ENGINEERING", "TEACHER"];
    (globalThis.fetch as any).mockResolvedValueOnce(jsonRes({ categories: cats }));
    expect(await fetchCategories()).toEqual(cats);
  });
  it("returns [] when 'categories' is absent (?? coalesce)", async () => {
    (globalThis.fetch as any).mockResolvedValueOnce(jsonRes({}));
    expect(await fetchCategories()).toEqual([]);
  });
  it("throws ApiError(500) on non-2xx with a fixed detail", async () => {
    (globalThis.fetch as any).mockResolvedValueOnce(jsonRes({ detail: "oops" }, { status: 500 }));
    await expect(fetchCategories()).rejects.toMatchObject({
      status: 500,
      detail: "Failed to fetch categories",
    });
  });
});

describe("rankPdfs", () => {
  beforeEach(() => vi.stubGlobal("fetch", vi.fn()));
  afterEach(() => vi.unstubAllGlobals());

  const okBody = {
    job_id: "j1",
    job_category: "ENGINEERING",
    total_candidates: 2,
    returned_candidates: 2,
    processing_time_seconds: 1.2,
    results: [],
  };

  it("POSTs multipart FormData to /api/rank/pdf with jd, category, files[]", async () => {
    const spy = globalThis.fetch as any;
    spy.mockResolvedValueOnce(jsonRes(okBody));
    const files = [new File(["a"], "r1.pdf"), new File(["b"], "r2.pdf")];

    await rankPdfs("a JD", "ENGINEERING", files);

    const [url, init] = spy.mock.calls[0];
    expect(url).toBe("/api/rank/pdf");
    expect(init.method).toBe("POST");
    const fd = init.body as FormData;
    expect(fd.get("job_description")).toBe("a JD");
    expect(fd.get("job_category")).toBe("ENGINEERING");
    expect(fd.getAll("files")).toHaveLength(2);
    expect((fd.getAll("files")[0] as File).name).toBe("r1.pdf");
  });

  it("returns the parsed ApiRankResponse on success", async () => {
    (globalThis.fetch as any).mockResolvedValueOnce(jsonRes(okBody));
    const res = await rankPdfs("jd", "ENGINEERING", [new File(["x"], "r.pdf")]);
    expect(res).toEqual(okBody);
  });

  it("unwraps body.detail into ApiError on non-2xx", async () => {
    (globalThis.fetch as any).mockResolvedValueOnce(jsonRes({ detail: "No valid PDFs" }, { status: 400 }));
    const p = rankPdfs("jd", "X", []);
    // Specifically an ApiError (not a plain Error), carrying the unwrapped detail
    await expect(p).rejects.toBeInstanceOf(ApiError);
    await expect(p).rejects.toMatchObject({ status: 400, detail: "No valid PDFs" });
  });

  it("falls back to 'HTTP <status>' when the error body is not JSON", async () => {
    (globalThis.fetch as any).mockResolvedValueOnce(new Response("not json", { status: 500 }));
    await expect(rankPdfs("jd", "X", [])).rejects.toMatchObject({
      status: 500,
      detail: "HTTP 500",
    });
  });
});
