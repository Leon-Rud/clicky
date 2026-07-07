# Clicky local bridge

A tiny local Node server that lets Clicky run on your **Claude subscription** instead of a pay-per-token Console API key.

It exposes an Anthropic-Messages-compatible endpoint on `http://127.0.0.1:8377` and forwards each request to the [Claude Agent SDK](https://www.npmjs.com/package/@anthropic-ai/claude-agent-sdk), which authenticates through the Claude Code login already on your Mac.

## Requirements

- Node.js 18+
- [Claude Code](https://code.claude.com/docs) installed **and logged in** on this Mac with a Claude subscription (run `claude` in a terminal and sign in once)

## Setup

```bash
cd local-bridge
npm install
npm start
```

You should see:

```
✅ Clicky local bridge listening on http://127.0.0.1:8377
```

**The bridge must be running whenever Clicky's "Claude Access" setting is "Claude subscription (local bridge)"** — the default. If the bridge is not running, Clicky shows an error telling you to start it. (In "API key" mode the bridge is not used at all.)

## Endpoints

| Route | Purpose |
|-------|---------|
| `GET /health` | Returns `200 {"status":"ok"}` — used to check the bridge is up |
| `POST /chat` | Accepts the standard Anthropic `/v1/messages` JSON body (model, max_tokens, stream, system, messages with base64 image + text blocks) and responds with Anthropic-format SSE (`stream: true`) or an Anthropic-format JSON message |

## How it maps requests to the Agent SDK

- The `system` string becomes the SDK's `systemPrompt`.
- The **latest user message's content blocks** — including base64 screenshot `image` blocks — are passed through unchanged via the SDK's streaming-input mode.
- Prior conversation turns are folded into a leading `<conversation_so_far>` transcript text block (the bridge is stateless; the app resends full history each request).
- The requested `model` (`claude-sonnet-*` / `claude-opus-*`) is passed straight through as the SDK's `model` option.
- All tools and file access are disabled (`tools: []`, `maxTurns: 1`) — this is pure vision chat.
- `max_tokens` is applied via the `CLAUDE_CODE_MAX_OUTPUT_TOKENS` environment variable (the Agent SDK has no per-request `max_tokens` option).
- Nothing is persisted: `persistSession: false` and no settings/CLAUDE.md loading (`settingSources: []`).

## Troubleshooting

- **Clicky says "Start the local bridge"** — run `cd local-bridge && npm start` and leave it running.
- **`bridge_error` mentioning authentication/login** — Claude Code isn't logged in. Run `claude` in a terminal and sign in with your Claude account.
- **Port 8377 already in use** — another bridge instance is probably running; that's fine, Clicky can use it.
