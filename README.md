# getbased-rag

A standalone RAG knowledge server — the backend that used to ship inside the getbased Electron desktop app. Now runs anywhere Docker runs, so you can point any client (the getbased PWA's *External server* lens backend, or your own) at it.

- **Stack**: FastAPI + Uvicorn · Qdrant (embedded local mode) · sentence-transformers / ONNX Runtime
- **Port**: 8322
- **Auth**: Bearer token, auto-generated at `/data/api_key` on first start
- **Stores**: every library is its own Qdrant collection under `/data/qdrant/`

---

## Quick start

```bash
docker compose up -d
docker compose exec lens lens key     # prints the API key you'll paste into the client
```

Server is now reachable at `http://127.0.0.1:8322`.

Smoke test:

```bash
KEY=$(docker compose exec -T lens lens key)
curl -s http://127.0.0.1:8322/health
curl -s -H "Authorization: Bearer $KEY" http://127.0.0.1:8322/stats
```

Ingest a file:

```bash
docker compose exec lens lens ingest /data/some-document.pdf
```

(put the file under the `lens-data` volume first — easiest way: `docker cp my.pdf getbased-rag:/data/`).

---

## Running without Docker Compose

```bash
docker run -d \
  --name getbased-rag \
  -p 127.0.0.1:8322:8322 \
  -v lens-data:/data \
  -e LENS_HOST=0.0.0.0 \
  ghcr.io/elkimek/getbased-rag:latest

docker exec getbased-rag lens key
```

---

## Wiring into the getbased PWA

In the PWA: **Settings → AI → Knowledge Base → External server**

| Field | Value |
|---|---|
| URL | `http://127.0.0.1:8322` |
| API key | the token printed by `lens key` |

Click **Save**, then **Test connection**. If the PWA reports `rag_ready: false`, that's expected — you haven't ingested any documents yet.

---

## Configuration

Every setting is driven by an environment variable. Defaults in parentheses.

| Variable | Purpose |
|---|---|
| `LENS_HOST` (`127.0.0.1`) | Interface to bind. In Docker this is overridden to `0.0.0.0` so the port mapping works |
| `LENS_PORT` (`8322`) | TCP port |
| `LENS_DATA_DIR` (`/data`) | Where the Qdrant DB, API key, and model cache live |
| `LENS_EMBEDDING_MODEL` (`sentence-transformers/all-MiniLM-L6-v2`) | HF model id |
| `LENS_SIMILARITY_FLOOR` (`0.55`) | Minimum cosine score for a chunk to be returned |
| `LENS_ONNX_PROVIDER` (auto) | `cuda` \| `rocm` \| `openvino` \| `coreml` \| `cpu` |
| `LENS_RERANKER` (`false`) | Enable reranking of top candidates |
| `LENS_CHUNK_MAX_SIZE` (`800`) | Max chunk size in characters |

The model is downloaded from HuggingFace on first query (~90 MB for MiniLM) and cached in the data volume.

### GPU acceleration

The `:latest` image ships the CPU provider. For CUDA:

1. Install the NVIDIA Container Toolkit on your host.
2. Run with `--gpus all` and `LENS_ONNX_PROVIDER=cuda`.

```bash
docker run -d --gpus all \
  -p 127.0.0.1:8322:8322 \
  -v lens-data:/data \
  -e LENS_ONNX_PROVIDER=cuda \
  ghcr.io/elkimek/getbased-rag:latest
```

For other providers (ROCm, DirectML, CoreML) you'll want a custom image that installs the matching `onnxruntime-*` wheel.

---

## HTTP API

All endpoints (except `/health` and `/`) require `Authorization: Bearer <key>`.

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/health` | Liveness + `rag_ready` + chunk count. Public |
| `POST` | `/query` | `{ query, top_k }` → top-k chunks from the active library |
| `GET` | `/stats` | Per-source chunk counts for the active library |
| `DELETE` | `/sources/{source}` | Drop one source from the active library |
| `DELETE` | `/sources` | Clear the active library |
| `GET` | `/libraries` | List libraries + active id |
| `POST` | `/libraries` | Create a new library |
| `POST` | `/libraries/{id}/activate` | Set active |
| `PATCH` | `/libraries/{id}` | Rename |
| `DELETE` | `/libraries/{id}` | Delete (drops Qdrant collection) |

---

## Security notes

- The default bind is `127.0.0.1` outside Docker — queries never leak to the LAN unless you explicitly set `LENS_HOST=0.0.0.0`.
- In Docker, the published port is pinned to `127.0.0.1:8322` in `docker-compose.yml`. If you want LAN access, change it to `8322:8322` — but then please also use a reverse proxy with TLS.
- The API key in `/data/api_key` is mode `0600` and never exposed over HTTP (use `lens key` locally to read it).

---

## Development (no Docker)

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[full]"
lens serve
```

In another terminal:

```bash
lens key
lens info
lens ingest ./some-directory
lens stats
```

---

## Roadmap

- [ ] CUDA / DirectML Docker variants
- [ ] Configurable embedding model tier (small / base / large) with guardrails
- [ ] OpenAPI-generated client for the PWA `external-server` backend
- [ ] Optional multi-tenant mode (per-owner API keys + library quotas)

---

## License

TBD — private repo during bring-up. Will land here once published.

---

## Lineage

This repo is the Python portion lifted out of [getbased](https://github.com/elkimek/getbased) after the Electron desktop app was retired. The PWA's `external-server` lens backend speaks this same HTTP contract unchanged.
