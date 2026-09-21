/**
 * Client for the standards-retrieval backend.
 *
 * Everything the UI knows about the live service goes through here, so the
 * rest of the app never deals with fetch, timeouts or error shapes.
 *
 * The backend is optional: if it is not running the UI stays usable and says
 * so, rather than showing a dead screen. It must never silently substitute
 * canned results for live ones — a demo that looks identical whether or not
 * the engine is running is worse than one that admits the engine is down.
 */

// Engine URL is hardcoded so the deployed frontend needs no environment
// variable. VITE_API_URL still overrides it for local development against
// a backend on another port.
//
// NOTE: this is a Cloudflare quick tunnel to a developer machine. It is a
// demo endpoint -- it dies when that machine sleeps and the hostname
// changes on every restart. Replace with a real container host before
// anyone depends on it.
const DEFAULT_ENGINE_URL = 'https://gmt-watt-lucy-walk.trycloudflare.com';
const BASE_URL = import.meta.env.VITE_API_URL ?? DEFAULT_ENGINE_URL;

/** Cold start loads two transformer models; first call is slow. */
const TIMEOUT_MS = 45000;

export class ApiError extends Error {
  constructor(message, { kind = 'unknown', status = null } = {}) {
    super(message);
    this.name = 'ApiError';
    this.kind = kind; // 'offline' | 'timeout' | 'http' | 'unknown'
    this.status = status;
  }
}

async function request(path, { method = 'GET', body, signal } = {}) {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), TIMEOUT_MS);

  // Abort if either the caller or our own timeout fires.
  if (signal) signal.addEventListener('abort', () => controller.abort(), { once: true });

  try {
    const res = await fetch(`${BASE_URL}${path}`, {
      method,
      headers: body ? { 'Content-Type': 'application/json' } : undefined,
      body: body ? JSON.stringify(body) : undefined,
      signal: controller.signal,
    });

    if (!res.ok) {
      let detail = `Request failed (${res.status})`;
      try {
        const payload = await res.json();
        if (payload?.detail) detail = payload.detail;
      } catch {
        /* non-JSON error body — keep the status-code message */
      }
      throw new ApiError(detail, { kind: 'http', status: res.status });
    }

    return await res.json();
  } catch (err) {
    if (err instanceof ApiError) throw err;

    if (err.name === 'AbortError') {
      // Caller-initiated aborts are not failures; let them propagate.
      if (signal?.aborted) throw err;
      throw new ApiError(
        'The engine took too long to respond. The first search after starting the backend loads the ranking models and can take up to a minute.',
        { kind: 'timeout' },
      );
    }

    // fetch() rejects with TypeError when it cannot reach the host at all.
    throw new ApiError(
      `Cannot reach the standards engine at ${BASE_URL}. Start the backend, then try again.`,
      { kind: 'offline' },
    );
  } finally {
    clearTimeout(timer);
  }
}

/**
 * Liveness plus what the engine is actually serving.
 * Returns null instead of throwing: callers poll this to show a status dot.
 */
export async function getHealth({ signal } = {}) {
  try {
    return await request('/health', { signal });
  } catch {
    return null;
  }
}

/**
 * Rank standards for a natural-language procurement query.
 *
 * Resolves to `{ query, results, confidence, confidence_reason, corpus_size }`
 * where `confidence` is 'strong' | 'uncertain' | 'none'. A 'none' verdict
 * means the corpus does not cover this query: results are nearest text
 * matches, not recommendations, and the UI must present them that way.
 */
export function retrieve(query, { topK = 10, language, explain = false, signal } = {}) {
  return request('/retrieve', {
    method: 'POST',
    body: { query, top_k: topK, language: language ?? null, explain },
    signal,
  });
}

/**
 * Languages the engine accepts queries in.
 *
 * Served by the backend rather than hardcoded here, so the list cannot drift
 * from what the translator actually supports. Falls back to English-only if
 * the backend is unreachable — the selector should not break the page.
 */
export async function listLanguages({ signal } = {}) {
  try {
    const data = await request('/languages', { signal });
    return data.languages ?? [];
  } catch {
    return [{ code: 'en', name: 'English', native: 'English' }];
  }
}

/** Files the backend can read. Mirrors SUPPORTED_EXTENSIONS in extraction.py. */
export const SUPPORTED_UPLOAD_TYPES = ['.pdf', '.docx', '.txt'];
export const MAX_UPLOAD_BYTES = 10 * 1024 * 1024;

/**
 * Upload a tender document, extract its specification, and search on it.
 *
 * Resolves to the extraction result with a nested `retrieval` holding the same
 * shape `retrieve()` returns. The extracted text comes back too, so the user
 * can check what was actually read rather than trusting an invisible step.
 */
export async function extractAndSearch(file, { topK = 10, signal } = {}) {
  if (file.size > MAX_UPLOAD_BYTES) {
    throw new ApiError(
      `That file is ${(file.size / 1048576).toFixed(1)} MB. The limit is ${MAX_UPLOAD_BYTES / 1048576} MB.`,
      { kind: 'http', status: 413 },
    );
  }

  const form = new FormData();
  form.append('file', file);

  // Deliberately not using request(): FormData must not get a JSON
  // Content-Type, and a large upload plus a cold model load needs longer.
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), 90000);
  if (signal) signal.addEventListener('abort', () => controller.abort(), { once: true });

  try {
    const res = await fetch(`${BASE_URL}/extract?top_k=${topK}`, {
      method: 'POST',
      body: form,
      signal: controller.signal,
    });

    if (!res.ok) {
      let detail = `Upload failed (${res.status})`;
      try {
        const payload = await res.json();
        if (payload?.detail) detail = payload.detail;
      } catch {
        /* non-JSON error body */
      }
      throw new ApiError(detail, { kind: 'http', status: res.status });
    }

    return await res.json();
  } catch (err) {
    if (err instanceof ApiError) throw err;
    if (err.name === 'AbortError') {
      if (signal?.aborted) throw err;
      throw new ApiError('Reading the document took too long.', { kind: 'timeout' });
    }
    throw new ApiError(
      `Cannot reach the standards engine at ${BASE_URL}. Start the backend, then try again.`,
      { kind: 'offline' },
    );
  } finally {
    clearTimeout(timer);
  }
}

/**
 * Every standard in the corpus, optionally filtered by sector.
 * The corpus is small enough to fetch whole; the catalogue filters client-side.
 */
export async function listStandards({ category, limit = 2000, signal } = {}) {
  // The engine returns a paginated envelope ({count, limit, offset, standards})
  // because the corpus is 6,383 records. Callers here expect a bare array, so
  // it is unwrapped at this boundary rather than de-paginating the API.
  const params = new URLSearchParams();
  if (category) params.set('category', category);
  params.set('limit', String(limit));
  const data = await request(`/standards?${params.toString()}`, { signal });
  if (Array.isArray(data)) return data;
  return Array.isArray(data?.standards) ? data.standards : [];
}

/**
 * One standard, by internal id (`IS-ELEC-009`) or IS number (`IS 694:2010`).
 * Throws an ApiError with `status === 404` when there is no such standard.
 */
export function getStandard(idOrNumber, { signal } = {}) {
  return request(`/standards/${encodeURIComponent(idOrNumber)}`, { signal });
}

/**
 * The allied-standards cluster around one standard.
 *
 * `researched: false` means no relationships have been recorded for it — not
 * that it has none. Entries flagged `outside_corpus` are real citations to
 * standards the pilot corpus does not hold; they are shown so the cluster is
 * not silently truncated, but they cannot be opened.
 */
export function getRelated(idOrNumber, { signal } = {}) {
  return request(`/standards/${encodeURIComponent(idOrNumber)}/related`, { signal });
}

/**
 * Published amendments for one standard.
 *
 * `checked: false` means the standard has not been researched — which is not
 * a statement that it has no amendments.
 */
export function getAmendments(idOrNumber, { signal } = {}) {
  return request(`/standards/${encodeURIComponent(idOrNumber)}/amendments`, { signal });
}

export { BASE_URL };
