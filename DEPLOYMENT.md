# Deployment

The frontend and the engine deploy separately, because they have
fundamentally different hosting requirements.

## Why not one Vercel deployment

A Vercel build of this repo failed with:

```
Error: Total bundle size (5675.91 MB) exceeds the maximum function size (500 MB).
```

That is not a configuration problem to trim around:

- `requirements.txt` pulls `torch`, `faiss-cpu`, `lightgbm`, `transformers`,
  `scipy` and `sklearn`. On Linux runners torch alone ships CUDA wheels of
  roughly 2.5 GB. The limit is 500 MB.
- Even with a trimmed dependency set the engine loads a **470 MB embedding
  model** and builds an index over **6,383 standards in ~50 s**, then answers
  from that index in memory.
- A serverless function keeps no memory between invocations, so every cold
  start would pay the full index build — well past any sane request timeout.

So: **Vercel hosts the React frontend. The engine runs in a container.**

## Frontend (Vercel)

`vercel.json` builds `frontend/` only; `.vercelignore` keeps the Python tree
out of the function bundle.

Set one environment variable in the Vercel project:

```
VITE_API_URL = https://<your-engine-host>
```

`frontend/src/api/client.js` reads it and falls back to
`http://localhost:8000` for local development.

## Engine (container)

`backend/Dockerfile` builds it, with the embedding model and the document
embeddings baked in at image build time so the first request is fast:

```bash
docker build -f backend/Dockerfile -t standards-engine .
docker run -p 8000:8000 \
  -e ALLOWED_ORIGINS=https://<your-vercel-domain> \
  -e GROQ_API_KEY=<optional, for LLM explanations> \
  standards-engine
```

Any container host works — Render, Railway, Fly.io, or a plain VM. Requirements:

| | |
|---|---|
| Memory | 2 GB minimum (model + index + reranker) |
| Disk | ~1.5 GB for the image |
| Probes | `/health/live` for liveness, `/health/ready` for readiness |
| Startup | ~10 s warm (baked embedding cache), ~60 s if the cache is cold |

Do **not** point a liveness probe at readiness: a liveness check that fails
during the index build restarts the container forever.

## Government deployment

Vercel plus a public container host is a demo posture, not a deployable one.
For actual use by officials the hosting itself is constrained:

- Hosting must be **MeitY-empanelled** (Government Community Cloud) or
  on-premises within India.
- A **CERT-In empanelled security audit** and "Safe to Host" certificate are
  required before go-live, and again after major changes.
- **STQC CQW certification** against GIGW 3.0, with a Website Quality Manual.
- The external LLM call (Groq) is permissible under DPDP 2023 today — it uses a
  negative list and no country is currently restricted — and this system only
  ever sends already-public standard metadata, never query text. A deploying
  department may still refuse any external dependency, so the explanation layer
  is optional and degrades to a deterministic template.

See the "Known limitations" section of `README.md` for what is not yet met.
