//
// Clicky local bridge server.
//
// Exposes an Anthropic-Messages-compatible endpoint on 127.0.0.1:8377 that is
// backed by the Claude Agent SDK instead of api.anthropic.com. The Agent SDK
// authenticates through the Claude Code login already present on this Mac, so
// requests draw from the user's Claude subscription — no pay-per-token API key.
//
// Routes:
//   GET  /health  → 200 {"status":"ok"} (used to check the bridge is running)
//   POST /chat    → accepts the standard Anthropic /v1/messages JSON body that
//                   ClaudeAPI.swift sends (model, max_tokens, stream, system,
//                   messages with base64 image + text content blocks) and
//                   responds either with Anthropic-format SSE (stream: true)
//                   or an Anthropic-format JSON message (stream omitted).
//

import http from "node:http";
import { randomUUID } from "node:crypto";
import { query } from "@anthropic-ai/claude-agent-sdk";

const BRIDGE_HOST = "127.0.0.1";
const BRIDGE_PORT = 8377;

// Screenshots arrive as base64 image blocks and can be several megabytes each.
const MAX_REQUEST_BODY_BYTES = 100 * 1024 * 1024;

/// Reads the full request body and parses it as JSON.
function readJSONRequestBody(request) {
    return new Promise((resolve, reject) => {
        const bodyChunks = [];
        let receivedByteCount = 0;

        request.on("data", (chunk) => {
            receivedByteCount += chunk.length;
            if (receivedByteCount > MAX_REQUEST_BODY_BYTES) {
                reject(new Error("Request body exceeds the maximum allowed size"));
                request.destroy();
                return;
            }
            bodyChunks.push(chunk);
        });
        request.on("end", () => {
            try {
                resolve(JSON.parse(Buffer.concat(bodyChunks).toString("utf8")));
            } catch (parseError) {
                reject(new Error(`Request body is not valid JSON: ${parseError.message}`));
            }
        });
        request.on("error", reject);
    });
}

/// The Anthropic API allows `system` to be a plain string or an array of text
/// blocks. ClaudeAPI.swift sends a string, but handle both shapes.
function extractSystemPromptText(systemField) {
    if (typeof systemField === "string") {
        return systemField;
    }
    if (Array.isArray(systemField)) {
        return systemField
            .filter((block) => block && block.type === "text" && typeof block.text === "string")
            .map((block) => block.text)
            .join("\n");
    }
    return "";
}

/// Renders a message's content (string or content-block array) as plain text.
/// Used only for folding prior conversation turns into the prompt — images in
/// history are represented by a short placeholder.
function renderMessageContentAsText(content) {
    if (typeof content === "string") {
        return content;
    }
    if (Array.isArray(content)) {
        return content
            .map((block) => {
                if (block && block.type === "text") return block.text;
                if (block && block.type === "image") return "[screenshot attached earlier]";
                return "";
            })
            .filter((text) => text.length > 0)
            .join("\n");
    }
    return "";
}

/// Builds the single Agent SDK user message for this request.
///
/// The bridge is stateless: ClaudeAPI.swift sends the full conversation on
/// every request (alternating user/assistant turns, then the latest user turn
/// with screenshot image blocks + text blocks). The Agent SDK's streaming
/// input only accepts *user* messages — assistant turns are produced by the
/// model — so prior turns are folded into a leading transcript text block,
/// and the latest user message's content blocks (base64 images included) are
/// passed through unchanged.
function buildAgentSDKUserMessage(anthropicMessages) {
    if (!Array.isArray(anthropicMessages) || anthropicMessages.length === 0) {
        throw new Error("Request body is missing a non-empty `messages` array");
    }

    const priorMessages = anthropicMessages.slice(0, -1);
    const latestMessage = anthropicMessages[anthropicMessages.length - 1];

    if (latestMessage.role !== "user") {
        throw new Error("The last message in `messages` must have role \"user\"");
    }

    const latestContentBlocks =
        typeof latestMessage.content === "string"
            ? [{ type: "text", text: latestMessage.content }]
            : latestMessage.content;

    const userContentBlocks = [];

    if (priorMessages.length > 0) {
        const transcriptLines = priorMessages.map((message) => {
            const speakerLabel = message.role === "assistant" ? "Assistant" : "User";
            return `${speakerLabel}: ${renderMessageContentAsText(message.content)}`;
        });
        userContentBlocks.push({
            type: "text",
            text:
                "<conversation_so_far>\n" +
                transcriptLines.join("\n") +
                "\n</conversation_so_far>\n" +
                "Continue the conversation. The current user turn follows.",
        });
    }

    userContentBlocks.push(...latestContentBlocks);

    // Exact SDKUserMessage shape required by the Agent SDK's streaming input
    // mode (verified against @anthropic-ai/claude-agent-sdk sdk.d.ts).
    return {
        type: "user",
        message: {
            role: "user",
            content: userContentBlocks,
        },
        parent_tool_use_id: null,
        session_id: "",
    };
}

/// Starts an Agent SDK query for one /chat request.
/// Tools and file access are fully disabled — this is pure vision chat.
function startAgentSDKQuery(requestBody) {
    const sdkUserMessage = buildAgentSDKUserMessage(requestBody.messages);

    async function* singleUserMessageStream() {
        yield sdkUserMessage;
    }

    const environmentForClaudeCode = { ...process.env };
    if (Number.isFinite(requestBody.max_tokens)) {
        // The Agent SDK has no per-request max_tokens option; Claude Code
        // honors this environment variable as the output-token cap instead.
        environmentForClaudeCode.CLAUDE_CODE_MAX_OUTPUT_TOKENS = String(requestBody.max_tokens);
    }

    return query({
        prompt: singleUserMessageStream(),
        options: {
            model: requestBody.model,
            systemPrompt: extractSystemPromptText(requestBody.system),
            // `tools: []` disables every built-in tool (Read, Bash, WebSearch, ...).
            tools: [],
            allowedTools: [],
            maxTurns: 1,
            includePartialMessages: true,
            // Don't write this throwaway exchange into ~/.claude session history.
            persistSession: false,
            // No filesystem settings/CLAUDE.md loading — full SDK isolation mode.
            settingSources: [],
            env: environmentForClaudeCode,
        },
    });
}

/// Writes one SSE event in the exact format the Anthropic API uses and that
/// ClaudeAPI.swift parses: an `event:` line followed by a `data: {json}` line.
function writeSSEEvent(response, eventPayload) {
    response.write(`event: ${eventPayload.type}\n`);
    response.write(`data: ${JSON.stringify(eventPayload)}\n\n`);
}

/// Synthesizes the full Anthropic SSE sequence for a complete piece of text.
/// Used as a fallback if the SDK finished without emitting stream events.
function writeSyntheticSSESequenceForText(response, model, fullText) {
    writeSSEEvent(response, {
        type: "message_start",
        message: {
            id: `msg_bridge_${randomUUID().replaceAll("-", "")}`,
            type: "message",
            role: "assistant",
            model: model,
            content: [],
            stop_reason: null,
            stop_sequence: null,
            usage: { input_tokens: 0, output_tokens: 0 },
        },
    });
    writeSSEEvent(response, {
        type: "content_block_start",
        index: 0,
        content_block: { type: "text", text: "" },
    });
    writeSSEEvent(response, {
        type: "content_block_delta",
        index: 0,
        delta: { type: "text_delta", text: fullText },
    });
    writeSSEEvent(response, { type: "content_block_stop", index: 0 });
    writeSSEEvent(response, {
        type: "message_delta",
        delta: { stop_reason: "end_turn", stop_sequence: null },
        usage: { output_tokens: 0 },
    });
    writeSSEEvent(response, { type: "message_stop" });
}

/// Handles POST /chat with "stream": true — Anthropic-format SSE.
async function handleStreamingChat(response, requestBody) {
    const agentQuery = startAgentSDKQuery(requestBody);

    response.writeHead(200, {
        "Content-Type": "text/event-stream",
        "Cache-Control": "no-cache",
        "Connection": "keep-alive",
    });

    let hasForwardedAnyStreamEvent = false;
    let hasForwardedMessageStop = false;
    let accumulatedAssistantText = "";
    let resultErrorDescription = null;

    try {
        for await (const sdkMessage of agentQuery) {
            if (sdkMessage.type === "stream_event") {
                // Raw Anthropic streaming event (message_start, content_block_start,
                // content_block_delta, content_block_stop, message_delta,
                // message_stop) — forward it verbatim.
                hasForwardedAnyStreamEvent = true;
                if (sdkMessage.event.type === "message_stop") {
                    hasForwardedMessageStop = true;
                }
                writeSSEEvent(response, sdkMessage.event);
            } else if (sdkMessage.type === "assistant") {
                // Full assistant message — kept only as a fallback source of text
                // in case no partial stream events were emitted.
                for (const contentBlock of sdkMessage.message.content ?? []) {
                    if (contentBlock.type === "text") {
                        accumulatedAssistantText += contentBlock.text;
                    }
                }
            } else if (sdkMessage.type === "result" && sdkMessage.subtype !== "success") {
                resultErrorDescription = `Agent SDK ended with ${sdkMessage.subtype}`;
            }
        }
    } catch (agentSDKError) {
        // The stream has already started, so an HTTP error status is no longer
        // possible — surface the failure as an SSE error event instead.
        writeSSEEvent(response, {
            type: "error",
            error: { type: "bridge_error", message: describeAgentSDKError(agentSDKError) },
        });
        response.write("data: [DONE]\n\n");
        response.end();
        return;
    }

    if (resultErrorDescription !== null && !hasForwardedAnyStreamEvent) {
        writeSSEEvent(response, {
            type: "error",
            error: { type: "bridge_error", message: resultErrorDescription },
        });
    } else if (!hasForwardedAnyStreamEvent) {
        // The SDK completed without partial events — synthesize the sequence
        // so the Swift client still receives the text.
        writeSyntheticSSESequenceForText(response, requestBody.model, accumulatedAssistantText);
        hasForwardedMessageStop = true;
    } else if (!hasForwardedMessageStop) {
        writeSSEEvent(response, { type: "message_stop" });
    }

    // ClaudeAPI.swift treats "data: [DONE]" as the end-of-stream marker.
    response.write("data: [DONE]\n\n");
    response.end();
}

/// Handles POST /chat without "stream": true — Anthropic-format JSON message.
async function handleNonStreamingChat(response, requestBody) {
    const agentQuery = startAgentSDKQuery(requestBody);

    let accumulatedAssistantText = "";
    let resultErrorDescription = null;

    for await (const sdkMessage of agentQuery) {
        if (sdkMessage.type === "assistant") {
            for (const contentBlock of sdkMessage.message.content ?? []) {
                if (contentBlock.type === "text") {
                    accumulatedAssistantText += contentBlock.text;
                }
            }
        } else if (sdkMessage.type === "result" && sdkMessage.subtype !== "success") {
            resultErrorDescription = `Agent SDK ended with ${sdkMessage.subtype}`;
        }
    }

    if (resultErrorDescription !== null && accumulatedAssistantText.length === 0) {
        writeJSONResponse(response, 502, {
            type: "error",
            error: { type: "bridge_error", message: resultErrorDescription },
        });
        return;
    }

    writeJSONResponse(response, 200, {
        id: `msg_bridge_${randomUUID().replaceAll("-", "")}`,
        type: "message",
        role: "assistant",
        model: requestBody.model,
        content: [{ type: "text", text: accumulatedAssistantText }],
        stop_reason: "end_turn",
        stop_sequence: null,
        usage: { input_tokens: 0, output_tokens: 0 },
    });
}

/// Produces a human-actionable description of an Agent SDK failure.
function describeAgentSDKError(agentSDKError) {
    const rawMessage = agentSDKError instanceof Error ? agentSDKError.message : String(agentSDKError);
    if (/login|logged in|authentication|credential|api key/i.test(rawMessage)) {
        return (
            `Claude Agent SDK authentication failed: ${rawMessage}. ` +
            "Make sure Claude Code is installed and logged in on this Mac (run `claude` in a terminal and sign in)."
        );
    }
    return `Claude Agent SDK request failed: ${rawMessage}`;
}

function writeJSONResponse(response, statusCode, jsonPayload) {
    const body = JSON.stringify(jsonPayload);
    response.writeHead(statusCode, {
        "Content-Type": "application/json",
        "Content-Length": Buffer.byteLength(body),
    });
    response.end(body);
}

const server = http.createServer(async (request, response) => {
    if (request.method === "GET" && request.url === "/health") {
        writeJSONResponse(response, 200, { status: "ok" });
        return;
    }

    if (request.method === "POST" && request.url === "/chat") {
        let requestBody;
        try {
            requestBody = await readJSONRequestBody(request);
        } catch (bodyError) {
            writeJSONResponse(response, 400, {
                type: "error",
                error: { type: "invalid_request_error", message: bodyError.message },
            });
            return;
        }

        try {
            if (requestBody.stream === true) {
                await handleStreamingChat(response, requestBody);
            } else {
                await handleNonStreamingChat(response, requestBody);
            }
        } catch (chatError) {
            if (!response.headersSent) {
                writeJSONResponse(response, 502, {
                    type: "error",
                    error: { type: "bridge_error", message: describeAgentSDKError(chatError) },
                });
            } else {
                response.end();
            }
            console.error("❌ /chat failed:", chatError);
        }
        return;
    }

    writeJSONResponse(response, 404, {
        type: "error",
        error: { type: "not_found_error", message: `No route for ${request.method} ${request.url}` },
    });
});

server.listen(BRIDGE_PORT, BRIDGE_HOST, () => {
    console.log(`✅ Clicky local bridge listening on http://${BRIDGE_HOST}:${BRIDGE_PORT}`);
    console.log("   POST /chat  — Anthropic-Messages-compatible endpoint backed by the Claude Agent SDK");
    console.log("   GET  /health — liveness check");
});
