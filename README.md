# lumina-mcp-router

**Progressive tool disclosure** MCP router with semantic search via embeddings.

A proxy that sits between an LLM agent and several MCP backends, exposing only **two meta-tools** (`search_tools`, `call_tool`) instead of the full catalogue. Typical context-window reduction on the tool schemas: **90–98%**.

## Why

The Model Context Protocol is awesome — until you plug 4 backends into your LLM and the tool definitions alone consume 20k+ tokens before the conversation even starts. Smaller / local models (Gemma, Qwen, Mistral) drown in options and start hallucinating tool names.

This router solves that by:

1. Connecting to all backends on startup and harvesting their `tools/list`.
2. Embedding every tool's `"{embedding_context}. Tool: {human tool name}. {description}"` via Ollama's `nomic-embed-text` (see [Semantic ranking](#semantic-ranking-embedding_context) below).
3. Exposing the LLM **only 2 meta-tools**:
   - `search_tools(query, top_k)` — semantic lookup returning the top-N candidate tools with their schemas.
   - `call_tool(name, arguments)` — invokes a specific backend tool by name.

The LLM first *searches* for what it needs, then *calls* it. Context stays tiny.

## Architecture

```
  Iris (Gemma) ──MCP SSE──▶ lumina-mcp-router ──MCP SSE──▶ gsuite backend
                                   │                   ──▶ microsoft backend
                                   │                   ──▶ odoo backend
                                   │                   ──▶ hass backend
                                   │
                                   └──HTTP──▶ Ollama (nomic-embed-text)
```

## Install (local dev)

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e '.[dev]'
pytest
```

## Run locally

```bash
export MCP_BACKENDS_CONFIG=./backends.yaml
export OLLAMA_BASE_URL=http://localhost:11434
python -m lumina_mcp_router
# → http://0.0.0.0:8080/sse (MCP SSE)
# → http://0.0.0.0:8080/health (admin)
# → http://0.0.0.0:8080/admin/reindex (POST)
```

Example `backends.yaml`:

```yaml
backends:
  - name: microsoft
    transport: sse              # optional, default: sse
    url: http://localhost:30201/sse
    embedding_context: "Microsoft 365 Outlook (Exchange, professional business email, Office 365, OneDrive, Teams, SharePoint)"
  - name: hass
    transport: sse
    url: http://localhost:30205/sse
    embedding_context: "Home Assistant (smart home, IoT devices, lights, switches, sensors, temperature, climate, automation)"
  - name: gsuite
    transport: streamablehttp   # HTTP + JSON streaming (MCP "Streamable HTTP")
    url: http://localhost:30203/mcp
    embedding_context: "Google Workspace (Gmail, Google Drive, Google Calendar, Google Docs, Google Sheets, personal @gmail.com accounts)"
  - name: odoo
    transport: streamablehttp
    url: http://localhost:30202/mcp
    embedding_context: "Odoo ERP (CRM, sales, invoicing, accounting, inventory, partners, customers, products, opportunities, leads)"
```

Each backend entry accepts:

| Field | Required | Default | Description |
|---|---|---|---|
| `name` | yes | — | Unique backend identifier, used as tool-name prefix (`<name>__<tool>`) |
| `url` | yes | — | Full URL to the backend MCP endpoint (path included) |
| `transport` | no | `sse` | Transport to use: `sse` or `streamablehttp` |
| `embedding_context` | no | *(backend name)* | Descriptive sentence injected into each tool's embedded text to sharpen semantic search (see below) |

## Semantic ranking (`embedding_context`)

The router embeds each tool with the following canonical text:

```
{embedding_context}. Tool: {human tool name}. {description}
```

Where:
- `embedding_context` is the per-backend string from `backends.yaml` (falls back to the backend name when unset).
- `human tool name` is the tool identifier with underscores replaced by spaces (e.g. `send_gmail_message` → `send gmail message`), so individual tokens contribute to the embedding.

**Why it matters.** Without a descriptive context, queries like *"read emails from Sourse inbox"* score Gmail (`gsuite`) and Outlook (`microsoft`) almost identically because both tool descriptions contain "email" and "inbox". The embedding model has no way to know that Sourse uses Microsoft 365.

By injecting a domain sentence — `"Microsoft 365 Outlook (Exchange, professional business email, @sourse.eu, ...)"` vs `"Google Workspace (Gmail, personal @gmail.com accounts, ...)"` — discriminating tokens (`Outlook`, `Exchange`, `Gmail`, ...) appear multiple times per tool, pulling the right backend to the top for queries that mention the concrete service or the relevant email domain.

**Before / after** for `gsuite.send_gmail_message`:

```
before: "[gsuite] send_gmail_message: Send an email via Gmail..."
after:  "Google Workspace (Gmail, Google Drive, Google Calendar, Google Docs, Google Sheets, personal @gmail.com accounts). Tool: send gmail message. Send an email via Gmail..."
```

And for `microsoft.search_emails`:

```
before: "[microsoft] search_emails: Search emails in the user's Outlook mailbox..."
after:  "Microsoft 365 Outlook (Exchange, professional business email, Office 365, OneDrive, Teams, SharePoint). Tool: search emails. Search emails in the user's Outlook mailbox..."
```

## Configuration

| Env var | Default | Description |
|---|---|---|
| `MCP_BACKENDS_CONFIG` | `/etc/lumina-mcp-router/backends.yaml` | Path to backends list |
| `OLLAMA_BASE_URL` | `http://ollama.ai.svc.cluster.local:11434` | Ollama HTTP endpoint |
| `EMBEDDING_MODEL` | `nomic-embed-text` | Ollama model to use |
| `LISTEN_HOST` | `0.0.0.0` | Bind address |
| `LISTEN_PORT` | `8080` | Bind port |
| `LOG_LEVEL` | `INFO` | Log level (DEBUG/INFO/WARN/ERROR) |
| `REINDEX_ENDPOINT_ENABLED` | `true` | Expose `POST /admin/reindex` |

## Deployment on Kubernetes

```bash
kubectl apply -f k8s/configmap.yaml
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml
```

The service is exposed as **NodePort 30210** (ClusterIP 8080 internal) in the `mcp-server` namespace.

Point your MCP client at:

- In-cluster DNS: `http://lumina-mcp-router.mcp-server.svc.cluster.local:8080/sse`
- NodePort (from the host): `http://<node-ip>:30210/sse`

## Example LLM flow

```text
user: "Send an email to bob@example.com with subject Hello"

LLM → search_tools(query="send an email to a recipient")
LLM ← {
  "results": [
    {
      "name": "gsuite__send_gmail_message",
      "backend": "gsuite",
      "description": "Send an email via Gmail...",
      "input_schema": { "type": "object", "properties": { "to": ..., "subject": ..., "body": ... } },
      "score": 0.87
    },
    ...
  ]
}

LLM → call_tool(name="gsuite__send_gmail_message",
                arguments={"to":"bob@example.com","subject":"Hello","body":"..."})
LLM ← { raw MCP result from the gsuite backend }
```

Tool names exposed by this router are always **prefixed by backend** (`<backend>__<original_name>`), so there's no ambiguity when multiple backends expose tools with similar names.

## Admin endpoints

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/health` | Health + indexed tool count |
| `GET` | `/tools` | List all indexed tools |
| `POST` | `/admin/reindex` | Reconnect backends and re-embed everything |

## License

MIT — see [LICENSE](LICENSE).
