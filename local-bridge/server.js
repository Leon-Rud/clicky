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
// Reliability behaviors (see the git history for the motivating bug):
//   - Every request is logged with timestamps, duration-to-first-token, and
//     outcome, so intermittent failures are visible in bridge.log.
//   - In streaming mode the SSE headers and a synthetic `message_start` event
//     are written the moment the request arrives, before the Agent SDK has
//     produced anything, so the Swift client never times out waiting for the
//     first response bytes while a slow SDK process spawns.
//   - Transient Agent SDK failures are retried once (with a fresh SDK process)
//     before an error is surfaced to the client.
//   - Clicky is a single-conversation app, so a new /chat request aborts any
//     /chat request that is still in flight (the user re-triggered push-to-talk).
//   - A warm single-use Claude Code process is kept ready between requests to
//     cut the 2-4s process-spawn latency. Each warm process serves exactly one
//     request (the bridge stays stateless — conversation history arrives in
//     every request body), so no context can leak between requests.
//

import http from "node:http";
import { randomUUID } from "node:crypto";
import { query } from "@anthropic-ai/claude-agent-sdk";

const BRIDGE_HOST = "127.0.0.1";
const BRIDGE_PORT = 8377;

// The Swift app's model picker pins the model generation that existed when the
// app was built. Upgrading here (instead of in Swift) avoids a rebuild, which
// would invalidate the user's TCC permission grants. Remove an entry to opt a
// model out of upgrading.
const MODEL_UPGRADE_MAP = {
    "claude-sonnet-4-6": "claude-sonnet-5",
    "claude-opus-4-6": "claude-opus-4-8",
};

// Screenshots arrive as base64 image blocks and can be several megabytes each.
const MAX_REQUEST_BODY_BYTES = 100 * 1024 * 1024;

// A warm Claude Code process older than this is discarded and respawned —
// long-idle child processes may have lost auth/network state.
const WARM_QUERY_MAX_AGE_MS = 10 * 60 * 1000;

// Only the most recent conversation turns are folded into the prompt. Older
// turns add prompt tokens and latency without helping a voice Q&A about the
// current screen. 12 messages = 6 user/assistant turn pairs.
const MAX_FOLDED_HISTORY_MESSAGES = 12;

// MARK: - Logging

let nextRequestLogNumber = 1;

/// Writes one timestamped line to stdout (nohup redirects stdout to bridge.log).
function logBridgeEvent(requestLogId, messageText) {
    const timestamp = new Date().toISOString();
    if (requestLogId) {
        console.log(`[${timestamp}] ${requestLogId} ${messageText}`);
    } else {
        console.log(`[${timestamp}] ${messageText}`);
    }
}

// MARK: - Request body parsing

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
                const parsedBody = JSON.parse(Buffer.concat(bodyChunks).toString("utf8"));
                parsedBody.__bodyByteCount = receivedByteCount;
                resolve(parsedBody);
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

// MARK: - Pointing reminder

/// Although the bridge passes the app's system prompt to the Agent SDK with
/// full fidelity (a plain-string `systemPrompt` REPLACES the Claude Code
/// preset — verified against sdk.mjs in @anthropic-ai/claude-agent-sdk), the
/// Claude Code harness still runs the request through its own agent loop and
/// can dilute instruction-following compared to a raw /v1/messages call. This
/// short reminder is appended to the user turn whenever the incoming system
/// prompt contains element-pointing instructions, restating the exact tag
/// syntax that CompanionManager.swift parses (see parsePointingCoordinates).
const POINTING_REMINDER_TEXT =
    "<pointing_reminder>\n" +
    "reminder: if a specific on-screen ui element (button, menu, field, area) is relevant to your answer, " +
    "you MUST end your response with a coordinate tag in exactly this format: [POINT:x,y:label] " +
    "— or [POINT:x,y:label:screenN] when the element is on a different screen than the cursor — " +
    "where x,y are integer pixel coordinates in the labeled screenshot's coordinate space (origin top-left) " +
    "and label is a short 1-3 word element name. err on the side of pointing. " +
    "if pointing genuinely wouldn't help, end with [POINT:none] instead. " +
    "if — and ONLY if — the user's request is explicitly an action command (they ask you to click, press, " +
    "open, or select something FOR them), use [CLICK:x,y:label] (or [CLICK:x,y:label:screenN]) instead of " +
    "POINT, and keep the spoken text to a brief confirmation like \"clicking the save button.\" " +
    "for informational questions always use POINT, and never emit both a POINT and a CLICK tag. " +
    "if the user's request is an action that needs MULTIPLE steps to complete (several clicks, or " +
    "clicking plus typing), speak a brief confirmation of the plan and end with [TASK] instead of a " +
    "CLICK tag — the step-by-step agent loop takes it from there. " +
    "the tag must be the very last thing in your response, after all spoken text.\n" +
    "</pointing_reminder>";

/// Detects whether the app's system prompt asks for [POINT:...] tags, so the
/// reminder is only injected for the pointing-enabled prompts.
function systemPromptRequestsPointing(systemPromptText) {
    return systemPromptText.includes("[POINT:");
}

// MARK: - Agent step reminder

/// Injected into the user turn for multi-step agent-task requests (system
/// prompts carrying the <agent_step> marker — see CompanionManager.swift's
/// agentStepSystemPrompt and runAgentTask). Restates the one-action-per-turn
/// protocol, which the Claude Code harness otherwise dilutes.
const AGENT_STEP_REMINDER_TEXT =
    "<agent_step_reminder>\n" +
    "reminder: respond with a tiny lowercase narration (two to six words) followed by EXACTLY ONE action tag " +
    "as the very last thing in your response: [CLICK:x,y:label], [TYPE:text], [KEY:combo], [SCROLL:up], " +
    "[SCROLL:down], [DONE:summary], or [FAIL:reason]. never two tags, never an action after the tag, " +
    "no markdown. coordinates are pixels in the labeled screenshot's coordinate space, origin top-left.\n" +
    "</agent_step_reminder>";

/// Detects the multi-step agent loop's step requests.
function systemPromptIsAgentStep(systemPromptText) {
    return systemPromptText.includes("<agent_step>");
}

// MARK: - Voice response style

/// Appended AFTER the app's own system prompt (never replacing it). Clicky
/// speaks every response aloud via ElevenLabs TTS, so response length equals
/// listening time — short plain-prose answers are the default. The coordinate
/// tag carve-out keeps this from overriding the [POINT:...] instructions: the
/// tag is stripped before TTS by the app, so it is not "spoken text".
const VOICE_RESPONSE_STYLE_INSTRUCTION =
    "\n\n<voice_response_style>\n" +
    "Your responses are spoken aloud to the user via text-to-speech, so response length equals listening time. " +
    "Default to 1-3 short conversational sentences. " +
    "Never use markdown: no asterisks, bullet lists, numbered lists, headings, or code blocks — plain spoken prose only. " +
    "Only give a longer answer when the user explicitly asks for detail. " +
    "This style rule applies to the spoken text only and does not override any coordinate-tag instructions above: " +
    "when those instructions require a [POINT:...] tag, you must still append it after the spoken text.\n" +
    "</voice_response_style>";

// MARK: - Agent SDK message + options construction

/// Builds the single Agent SDK user message for this request.
///
/// The bridge is stateless: ClaudeAPI.swift sends the full conversation on
/// every request (alternating user/assistant turns, then the latest user turn
/// with screenshot image blocks + text blocks). The Agent SDK's streaming
/// input only accepts *user* messages — assistant turns are produced by the
/// model — so prior turns are folded into a leading transcript text block,
/// and the latest user message's content blocks (base64 images included) are
/// passed through unchanged.
function buildAgentSDKUserMessage(anthropicMessages, systemPromptText, requestLogId) {
    if (!Array.isArray(anthropicMessages) || anthropicMessages.length === 0) {
        throw new Error("Request body is missing a non-empty `messages` array");
    }

    const allPriorMessages = anthropicMessages.slice(0, -1);
    // Cap the folded history at the most recent turns — see
    // MAX_FOLDED_HISTORY_MESSAGES for why older turns are dropped.
    const historyWasTruncated = allPriorMessages.length > MAX_FOLDED_HISTORY_MESSAGES;
    const priorMessages = historyWasTruncated
        ? allPriorMessages.slice(-MAX_FOLDED_HISTORY_MESSAGES)
        : allPriorMessages;
    logBridgeEvent(
        requestLogId ?? null,
        `folded history: kept ${priorMessages.length} of ${allPriorMessages.length} prior messages` +
            (historyWasTruncated ? " (older turns truncated)" : "")
    );
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
        if (historyWasTruncated) {
            transcriptLines.unshift("(earlier conversation omitted)");
        }
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

    // Reinforce the [POINT:...] instructions inside the user turn — the
    // system prompt alone under-triggers pointing through the Agent SDK.
    // Agent-step requests get their own one-action-per-turn reminder instead.
    if (systemPromptIsAgentStep(systemPromptText)) {
        userContentBlocks.push({ type: "text", text: AGENT_STEP_REMINDER_TEXT });
    } else if (systemPromptRequestsPointing(systemPromptText)) {
        userContentBlocks.push({ type: "text", text: POINTING_REMINDER_TEXT });
    }

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

/// Builds the Agent SDK options for a request (everything except the prompt).
/// Tools and file access are fully disabled — this is pure vision chat.
function buildAgentSDKOptions(requestBody, abortController) {
    const environmentForClaudeCode = { ...process.env };
    if (Number.isFinite(requestBody.max_tokens)) {
        // The Agent SDK has no per-request max_tokens option; Claude Code
        // honors this environment variable as the output-token cap instead.
        environmentForClaudeCode.CLAUDE_CODE_MAX_OUTPUT_TOKENS = String(requestBody.max_tokens);
    }

    const systemPromptText = extractSystemPromptText(requestBody.system);
    // Locator requests (ElementLocationDetector.swift's precision coordinate
    // pass) demand a strict-JSON answer, so the spoken-voice style instruction
    // must NOT be appended — it would push the model back to prose. Agent-step
    // requests fully specify their own narration style, so the voice-style
    // instruction is skipped for them too.
    const isLocatorRequest = systemPromptText.includes("<locator_request>");
    const isAgentStepRequest = systemPromptIsAgentStep(systemPromptText);

    return {
        model: requestBody.model,
        // A plain string REPLACES Claude Code's preset system prompt entirely
        // (maximum fidelity) — this is NOT the `{type:'preset', append}` mode.
        // The voice-style instruction is appended AFTER the app's own prompt
        // so it never displaces the app's instructions (including pointing).
        systemPrompt: (isLocatorRequest || isAgentStepRequest)
            ? systemPromptText
            : systemPromptText + VOICE_RESPONSE_STYLE_INSTRUCTION,
        // Pin extended thinking off. Clicky is a look-at-screenshot-and-answer
        // app where time-to-first-token matters more than deep reasoning. In
        // SDK mode Claude Code already defaults the thinking budget to 0, but
        // an explicit 0 makes that deterministic — it also wins over the
        // MAX_THINKING_TOKENS env var and the "ultrathink" keyword trigger
        // that would otherwise enable a large thinking budget.
        // (Verified against sdk.mjs: maxThinkingTokens → --max-thinking-tokens.)
        maxThinkingTokens: 0,
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
        abortController,
    };
}

// MARK: - Warm single-use Claude Code process

/// The Agent SDK spawns and initializes its Claude Code child process as soon
/// as query() is called, even though the streaming-input generator has not
/// yielded a user message yet. The bridge exploits this to keep ONE pre-spawned
/// process ready between requests: the generator awaits a deferred promise,
/// and when the next /chat arrives with matching options the message is
/// delivered and the 2-4s spawn cost has already been paid.
///
/// A warm process serves exactly one request and then exits (the generator
/// ends after one message, `maxTurns: 1`, `persistSession: false`), so no
/// conversation context can leak between requests.
let warmAgentQuerySlot = null;
// The (model, system, max_tokens) template of the most recent request —
// used to pre-spawn the next warm process with matching options.
let lastChatRequestTemplate = null;

/// The identity key for a warm process. All Agent SDK options that are fixed
/// at spawn time and vary between Clicky requests must be part of this key.
function computeWarmQueryKey(requestBody) {
    return JSON.stringify({
        model: requestBody.model,
        systemPromptText: extractSystemPromptText(requestBody.system),
        maxTokens: Number.isFinite(requestBody.max_tokens) ? requestBody.max_tokens : null,
    });
}

/// Spawns a warm Claude Code process for the given request template.
function createWarmAgentQuerySlot(requestTemplate) {
    try {
        const abortController = new AbortController();
        let deliverUserMessage;
        const pendingUserMessage = new Promise((resolve) => {
            deliverUserMessage = resolve;
        });

        async function* warmUserMessageStream() {
            yield await pendingUserMessage;
        }

        const agentQuery = query({
            prompt: warmUserMessageStream(),
            options: buildAgentSDKOptions(requestTemplate, abortController),
        });

        return {
            optionsKey: computeWarmQueryKey(requestTemplate),
            agentQuery,
            deliverUserMessage,
            abortController,
            createdAt: Date.now(),
        };
    } catch (spawnError) {
        logBridgeEvent(null, `⚠️ failed to pre-spawn warm SDK process: ${spawnError.message}`);
        return null;
    }
}

/// Aborts and clears the current warm process (options mismatch / staleness).
function discardWarmAgentQuerySlot(reason) {
    if (warmAgentQuerySlot === null) return;
    logBridgeEvent(null, `♻️ discarding warm SDK process (${reason})`);
    try {
        warmAgentQuerySlot.abortController.abort();
    } catch {
        // The child process may already be gone — nothing to do.
    }
    warmAgentQuerySlot = null;
}

/// Pre-spawns the next warm process after a request completes, if none is
/// already waiting. Runs on a microtask so it never delays the response.
function replenishWarmAgentQuerySlot() {
    if (warmAgentQuerySlot !== null || lastChatRequestTemplate === null) return;
    warmAgentQuerySlot = createWarmAgentQuerySlot(lastChatRequestTemplate);
    if (warmAgentQuerySlot !== null) {
        logBridgeEvent(null, "🔥 warm SDK process ready for next request");
    }
}

/// Returns an Agent SDK query for this request — the warm process when its
/// options match, otherwise a cold spawn — plus that query's abort controller.
///
/// `mayUseWarmProcess: false` bypasses the warm slot entirely (never consumes
/// or discards it). Used for non-streaming locator requests so they can't
/// evict the warm process that the next voice request depends on.
function acquireAgentQueryForRequest(requestBody, sdkUserMessage, mayUseWarmProcess = true) {
    const requestOptionsKey = computeWarmQueryKey(requestBody);

    if (mayUseWarmProcess && warmAgentQuerySlot !== null) {
        const warmSlot = warmAgentQuerySlot;
        const warmSlotAgeMs = Date.now() - warmSlot.createdAt;
        if (warmSlot.optionsKey === requestOptionsKey && warmSlotAgeMs < WARM_QUERY_MAX_AGE_MS) {
            warmAgentQuerySlot = null;
            warmSlot.deliverUserMessage(sdkUserMessage);
            return { agentQuery: warmSlot.agentQuery, abortController: warmSlot.abortController, wasWarm: true };
        }
        discardWarmAgentQuerySlot(
            warmSlot.optionsKey === requestOptionsKey ? `older than ${WARM_QUERY_MAX_AGE_MS / 1000}s` : "options changed"
        );
    }

    const abortController = new AbortController();
    async function* singleUserMessageStream() {
        yield sdkUserMessage;
    }
    const agentQuery = query({
        prompt: singleUserMessageStream(),
        options: buildAgentSDKOptions(requestBody, abortController),
    });
    return { agentQuery, abortController, wasWarm: false };
}

// MARK: - Single-conversation concurrency

/// Clicky is single-conversation: when a new /chat arrives while a previous
/// one is still streaming (the user re-triggered push-to-talk), the previous
/// request's SDK process is aborted so two Claude Code processes never race
/// each other for auth/config state.
let activeChatRequestContext = null;

function registerActiveChatRequest(chatRequestContext) {
    if (activeChatRequestContext !== null) {
        logBridgeEvent(
            chatRequestContext.requestLogId,
            `⛔ superseding in-flight request ${activeChatRequestContext.requestLogId}`
        );
        activeChatRequestContext.supersede();
    }
    activeChatRequestContext = chatRequestContext;
}

function unregisterActiveChatRequest(chatRequestContext) {
    if (activeChatRequestContext === chatRequestContext) {
        activeChatRequestContext = null;
    }
}

// MARK: - SSE writing

/// Writes one SSE event in the exact format the Anthropic API uses and that
/// ClaudeAPI.swift parses: an `event:` line followed by a `data: {json}` line.
function writeSSEEvent(response, eventPayload) {
    if (response.writableEnded) return;
    response.write(`event: ${eventPayload.type}\n`);
    response.write(`data: ${JSON.stringify(eventPayload)}\n\n`);
}

/// Writes the synthetic `message_start` that is sent the moment a streaming
/// request arrives — before the Agent SDK has produced anything — so the
/// Swift client receives response bytes immediately and never times out
/// waiting for a slow SDK process to spawn. ClaudeAPI.swift only acts on
/// `content_block_delta` events, so an early message_start followed by a gap
/// (and by the SDK's own suppressed message_start) is harmless.
function writeImmediateSyntheticMessageStart(response, model) {
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
}

/// Synthesizes the SSE body events for a complete piece of text (everything
/// after message_start, which has already been written). Used as a fallback
/// if the SDK finished without emitting partial stream events.
function writeSyntheticSSEBodyForText(response, fullText) {
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

// MARK: - /chat handlers

/// Handles POST /chat with "stream": true — Anthropic-format SSE.
async function handleStreamingChat(response, requestBody, chatRequestContext) {
    const { requestLogId, requestStartedAt } = chatRequestContext;

    // Build the SDK message first — if the request body is malformed this
    // throws before headers are sent, so the client still gets a JSON error.
    const sdkUserMessage = buildAgentSDKUserMessage(
        requestBody.messages,
        extractSystemPromptText(requestBody.system),
        requestLogId
    );

    // Send headers + message_start IMMEDIATELY so the Swift client always has
    // response bytes within milliseconds, regardless of SDK spawn latency.
    response.writeHead(200, {
        "Content-Type": "text/event-stream",
        "Cache-Control": "no-cache",
        "Connection": "keep-alive",
    });
    writeImmediateSyntheticMessageStart(response, requestBody.model);

    let hasForwardedContentEvent = false;
    let hasForwardedMessageStop = false;
    let firstTokenElapsedMs = null;
    let accumulatedAssistantText = "";

    const maxAttemptCount = 2;
    for (let attemptNumber = 1; attemptNumber <= maxAttemptCount; attemptNumber++) {
        let attemptErrorDescription = null;
        accumulatedAssistantText = "";

        try {
            const { agentQuery, abortController, wasWarm } = acquireAgentQueryForRequest(requestBody, sdkUserMessage);
            chatRequestContext.currentAbortController = abortController;
            logBridgeEvent(requestLogId, `attempt ${attemptNumber}/${maxAttemptCount} using ${wasWarm ? "warm" : "cold"} SDK process`);

            for await (const sdkMessage of agentQuery) {
                if (chatRequestContext.superseded || chatRequestContext.clientDisconnected) break;

                if (sdkMessage.type === "stream_event") {
                    const streamEvent = sdkMessage.event;
                    // A synthetic message_start was already written up front —
                    // drop the SDK's own so the client sees exactly one.
                    if (streamEvent.type === "message_start") continue;
                    if (streamEvent.type === "message_stop") {
                        hasForwardedMessageStop = true;
                    }
                    if (streamEvent.type.startsWith("content_block")) {
                        if (firstTokenElapsedMs === null) {
                            firstTokenElapsedMs = Date.now() - requestStartedAt;
                            logBridgeEvent(requestLogId, `first token after ${(firstTokenElapsedMs / 1000).toFixed(2)}s`);
                        }
                        hasForwardedContentEvent = true;
                        if (
                            streamEvent.type === "content_block_delta" &&
                            streamEvent.delta?.type === "text_delta" &&
                            typeof streamEvent.delta.text === "string"
                        ) {
                            accumulatedAssistantText += streamEvent.delta.text;
                        }
                    }
                    writeSSEEvent(response, streamEvent);
                } else if (sdkMessage.type === "assistant") {
                    // Full assistant message — kept only as a fallback source of
                    // text in case no partial stream events were emitted.
                    if (!hasForwardedContentEvent) {
                        for (const contentBlock of sdkMessage.message.content ?? []) {
                            if (contentBlock.type === "text") {
                                accumulatedAssistantText += contentBlock.text;
                            }
                        }
                    }
                } else if (sdkMessage.type === "result" && sdkMessage.subtype !== "success") {
                    attemptErrorDescription = `Agent SDK ended with ${sdkMessage.subtype}`;
                }
            }
        } catch (agentSDKError) {
            attemptErrorDescription = describeAgentSDKError(agentSDKError);
        }

        if (chatRequestContext.superseded || chatRequestContext.clientDisconnected) {
            const abandonReason = chatRequestContext.superseded ? "superseded by newer request" : "client disconnected";
            logBridgeEvent(requestLogId, `abandoned (${abandonReason}) after ${elapsedSecondsText(requestStartedAt)}`);
            if (!response.writableEnded) {
                response.write("data: [DONE]\n\n");
                response.end();
            }
            return;
        }

        if (attemptErrorDescription === null) {
            // Success. If the SDK never emitted partial events, synthesize the
            // body from the accumulated assistant text so the client still
            // receives the response.
            if (!hasForwardedContentEvent) {
                writeSyntheticSSEBodyForText(response, accumulatedAssistantText);
            } else if (!hasForwardedMessageStop) {
                writeSSEEvent(response, { type: "message_stop" });
            }
            const containsPointTag = /\[(?:POINT|CLICK):/.test(accumulatedAssistantText);
            logBridgeEvent(
                requestLogId,
                `✅ ok in ${elapsedSecondsText(requestStartedAt)} ` +
                    `(first token ${firstTokenElapsedMs !== null ? (firstTokenElapsedMs / 1000).toFixed(2) + "s" : "n/a"}, ` +
                    `${accumulatedAssistantText.length} chars, point tag: ${containsPointTag ? "yes" : "no"})`
            );
            break;
        }

        // Failure. Retry once with a fresh SDK process — but only if nothing
        // has been forwarded to the client yet (a clean restart is possible).
        if (!hasForwardedContentEvent && attemptNumber < maxAttemptCount) {
            logBridgeEvent(requestLogId, `⚠️ attempt ${attemptNumber} failed (${attemptErrorDescription}) — retrying with a fresh SDK process`);
            continue;
        }

        // The stream has already started (200 + message_start were written),
        // so an HTTP error status is no longer possible — surface the failure
        // as an SSE error event instead.
        writeSSEEvent(response, {
            type: "error",
            error: { type: "bridge_error", message: attemptErrorDescription },
        });
        logBridgeEvent(requestLogId, `❌ failed after ${elapsedSecondsText(requestStartedAt)}: ${attemptErrorDescription}`);
        break;
    }

    // ClaudeAPI.swift treats "data: [DONE]" as the end-of-stream marker.
    if (!response.writableEnded) {
        response.write("data: [DONE]\n\n");
        response.end();
    }
}

/// Handles POST /chat without "stream": true — Anthropic-format JSON message.
async function handleNonStreamingChat(response, requestBody, chatRequestContext) {
    const { requestLogId, requestStartedAt } = chatRequestContext;

    const sdkUserMessage = buildAgentSDKUserMessage(
        requestBody.messages,
        extractSystemPromptText(requestBody.system),
        requestLogId
    );

    let accumulatedAssistantText = "";
    let finalErrorDescription = null;

    const maxAttemptCount = 2;
    for (let attemptNumber = 1; attemptNumber <= maxAttemptCount; attemptNumber++) {
        accumulatedAssistantText = "";
        let attemptErrorDescription = null;

        try {
            // Non-streaming requests are locator calls — keep them away from
            // the warm slot so the main voice flow's pre-spawned process survives.
            const { agentQuery, abortController, wasWarm } = acquireAgentQueryForRequest(requestBody, sdkUserMessage, false);
            chatRequestContext.currentAbortController = abortController;
            logBridgeEvent(requestLogId, `attempt ${attemptNumber}/${maxAttemptCount} using ${wasWarm ? "warm" : "cold"} SDK process`);

            for await (const sdkMessage of agentQuery) {
                if (chatRequestContext.superseded || chatRequestContext.clientDisconnected) break;
                if (sdkMessage.type === "assistant") {
                    for (const contentBlock of sdkMessage.message.content ?? []) {
                        if (contentBlock.type === "text") {
                            accumulatedAssistantText += contentBlock.text;
                        }
                    }
                } else if (sdkMessage.type === "result" && sdkMessage.subtype !== "success") {
                    attemptErrorDescription = `Agent SDK ended with ${sdkMessage.subtype}`;
                }
            }
        } catch (agentSDKError) {
            attemptErrorDescription = describeAgentSDKError(agentSDKError);
        }

        if (chatRequestContext.superseded || chatRequestContext.clientDisconnected) {
            logBridgeEvent(requestLogId, `abandoned after ${elapsedSecondsText(requestStartedAt)}`);
            if (!response.writableEnded) response.end();
            return;
        }

        if (attemptErrorDescription === null || accumulatedAssistantText.length > 0) {
            finalErrorDescription = null;
            break;
        }

        finalErrorDescription = attemptErrorDescription;
        if (attemptNumber < maxAttemptCount) {
            logBridgeEvent(requestLogId, `⚠️ attempt ${attemptNumber} failed (${attemptErrorDescription}) — retrying with a fresh SDK process`);
        }
    }

    if (finalErrorDescription !== null) {
        logBridgeEvent(requestLogId, `❌ failed after ${elapsedSecondsText(requestStartedAt)}: ${finalErrorDescription}`);
        writeJSONResponse(response, 502, {
            type: "error",
            error: { type: "bridge_error", message: finalErrorDescription },
        });
        return;
    }

    logBridgeEvent(
        requestLogId,
        `✅ ok in ${elapsedSecondsText(requestStartedAt)} (${accumulatedAssistantText.length} chars, ` +
            `point tag: ${/\[(?:POINT|CLICK):/.test(accumulatedAssistantText) ? "yes" : "no"})`
    );
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

// MARK: - Helpers

function elapsedSecondsText(startedAtMs) {
    return `${((Date.now() - startedAtMs) / 1000).toFixed(2)}s`;
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
    if (response.writableEnded || response.headersSent) return;
    const body = JSON.stringify(jsonPayload);
    response.writeHead(statusCode, {
        "Content-Type": "application/json",
        "Content-Length": Buffer.byteLength(body),
    });
    response.end(body);
}

/// Counts image content blocks in the latest user message (for logging).
function countImagesInLatestMessage(anthropicMessages) {
    if (!Array.isArray(anthropicMessages) || anthropicMessages.length === 0) return 0;
    const latestContent = anthropicMessages[anthropicMessages.length - 1].content;
    if (!Array.isArray(latestContent)) return 0;
    return latestContent.filter((block) => block && block.type === "image").length;
}

// MARK: - HTTP server

const server = http.createServer(async (request, response) => {
    if (request.method === "GET" && request.url === "/health") {
        writeJSONResponse(response, 200, {
            status: "ok",
            warmProcessReady: warmAgentQuerySlot !== null,
        });
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

        // The Swift app pins older model IDs; upgrade them to the current
        // generation here so the app gets smarter answers without a rebuild
        // (rebuilding would invalidate the user's TCC permission grants).
        const upgradedModel = MODEL_UPGRADE_MAP[requestBody.model];
        if (upgradedModel !== undefined) {
            requestBody.__originalRequestedModel = requestBody.model;
            requestBody.model = upgradedModel;
        }

        const chatRequestContext = {
            requestLogId: `#${nextRequestLogNumber++}`,
            requestStartedAt: Date.now(),
            superseded: false,
            clientDisconnected: false,
            currentAbortController: null,
            supersede() {
                this.superseded = true;
                this.currentAbortController?.abort();
            },
        };

        const bodyMegabytes = ((requestBody.__bodyByteCount ?? 0) / 1_048_576).toFixed(1);
        logBridgeEvent(
            chatRequestContext.requestLogId,
            `POST /chat ${requestBody.stream === true ? "stream" : "json"} model=${requestBody.model} ` +
                `body=${bodyMegabytes}MB images=${countImagesInLatestMessage(requestBody.messages)} ` +
                `messages=${Array.isArray(requestBody.messages) ? requestBody.messages.length : 0}`
        );

        // Abort any /chat still in flight — Clicky is single-conversation.
        registerActiveChatRequest(chatRequestContext);

        // If the Swift client gives up (task cancelled / app quit), stop the
        // SDK process instead of letting it run against a dead socket.
        response.on("close", () => {
            if (!response.writableEnded) {
                chatRequestContext.clientDisconnected = true;
                chatRequestContext.currentAbortController?.abort();
            }
        });

        // Remember this request's options so the next warm process matches.
        // Only streaming requests (the main voice flow) set the template —
        // non-streaming locator calls must not retarget the warm process.
        if (requestBody.stream === true) {
            lastChatRequestTemplate = {
                model: requestBody.model,
                system: requestBody.system,
                max_tokens: requestBody.max_tokens,
            };
        }

        try {
            if (requestBody.stream === true) {
                await handleStreamingChat(response, requestBody, chatRequestContext);
            } else {
                await handleNonStreamingChat(response, requestBody, chatRequestContext);
            }
        } catch (chatError) {
            if (!response.headersSent) {
                writeJSONResponse(response, 502, {
                    type: "error",
                    error: { type: "bridge_error", message: describeAgentSDKError(chatError) },
                });
            } else if (!response.writableEnded) {
                response.end();
            }
            logBridgeEvent(chatRequestContext.requestLogId, `❌ /chat handler threw: ${chatError.stack ?? chatError}`);
        } finally {
            unregisterActiveChatRequest(chatRequestContext);
            // Pre-spawn the next warm SDK process off the response path.
            setImmediate(replenishWarmAgentQuerySlot);
        }
        return;
    }

    writeJSONResponse(response, 404, {
        type: "error",
        error: { type: "not_found_error", message: `No route for ${request.method} ${request.url}` },
    });
});

server.listen(BRIDGE_PORT, BRIDGE_HOST, () => {
    logBridgeEvent(null, `✅ Clicky local bridge listening on http://${BRIDGE_HOST}:${BRIDGE_PORT}`);
    logBridgeEvent(null, "   POST /chat  — Anthropic-Messages-compatible endpoint backed by the Claude Agent SDK");
    logBridgeEvent(null, "   GET  /health — liveness check");
});
