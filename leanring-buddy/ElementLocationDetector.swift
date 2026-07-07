//
//  ElementLocationDetector.swift
//  leanring-buddy
//
//  Precision second pass for element pointing and click actions. The main
//  Claude response includes a rough [POINT:...]/[CLICK:...] coordinate; this
//  detector re-asks Claude (through the local bridge, on the user's
//  subscription) to locate the exact element in the SAME screenshot and
//  return refined coordinates as strict JSON. On any failure — bridge down,
//  timeout, unparseable reply, element not found — callers fall back to the
//  original tag coordinates.
//

import AppKit
import Foundation

/// Refines a rough element coordinate from the main response into an exact one
/// by sending the screenshot plus the element label to Claude via the local
/// bridge (`http://127.0.0.1:8377/chat`, non-streaming). Uses a plain vision
/// prompt that demands a strict JSON answer — no Computer Use beta/tooling,
/// because the bridge's Agent SDK backend can't pass tool definitions through.
class ElementLocationDetector {

    /// The local bridge's chat endpoint. The bridge authenticates via the
    /// Claude Agent SDK on this machine, so no API key is involved.
    private static let localBridgeChatURLString = "http://127.0.0.1:8377/chat"

    /// A locator call must be fast — it sits between the main response and the
    /// pointing/click animation. Past this budget the caller uses the original
    /// tag coordinates instead, so a slow refinement never blocks the user.
    private static let locatorTimeoutSeconds: TimeInterval = 6

    /// The "<locator_request>" marker tells the bridge this is a locator call:
    /// the bridge skips its spoken-voice style instruction (which would tell
    /// the model to answer in conversational prose instead of strict JSON).
    private static let locatorSystemPrompt = """
    <locator_request>
    You are a precision UI element locator. You receive one screenshot and the name of a UI element to find. \
    Respond with ONLY a strict JSON object and nothing else — no prose, no markdown fences, no explanation. \
    Format: {"x":123,"y":456,"found":true} where x and y are integer pixel coordinates of the CENTER of the \
    element in the screenshot's coordinate space (origin top-left, x increases rightward, y increases downward). \
    If the element is not visible in the screenshot, respond with {"x":0,"y":0,"found":false}.
    """

    private let localBridgeChatURL: URL
    private let model: String
    private let session: URLSession

    init(model: String = "claude-sonnet-4-6") {
        self.localBridgeChatURL = URL(string: Self.localBridgeChatURLString)!
        self.model = model

        let config = URLSessionConfiguration.default
        config.timeoutIntervalForRequest = Self.locatorTimeoutSeconds
        config.timeoutIntervalForResource = Self.locatorTimeoutSeconds
        config.waitsForConnectivity = false
        config.urlCache = nil
        config.httpCookieStorage = nil
        self.session = URLSession(configuration: config)
    }

    /// Asks Claude to locate the exact center of the labeled element in the
    /// screenshot that was already sent with the main request.
    ///
    /// - Parameters:
    ///   - screenshotData: The SAME JPEG screenshot used for the main request
    ///     (already downscaled to the API-safe resolution — do not recapture).
    ///   - screenshotWidthInPixels: The screenshot's width in pixels.
    ///   - screenshotHeightInPixels: The screenshot's height in pixels.
    ///   - elementLabel: The element label from the [POINT:...]/[CLICK:...] tag
    ///     (e.g. "save button").
    ///   - initialCoordinate: The rough coordinate from the tag, in the
    ///     screenshot's pixel space — given to the model as a starting estimate.
    ///
    /// - Returns: The refined coordinate in the screenshot's pixel space
    ///   (top-left origin), or `nil` on any failure so the caller falls back
    ///   to `initialCoordinate`.
    func refineElementCoordinate(
        screenshotData: Data,
        screenshotWidthInPixels: Int,
        screenshotHeightInPixels: Int,
        elementLabel: String,
        initialCoordinate: CGPoint
    ) async -> CGPoint? {
        let userPrompt = """
        The screenshot is \(screenshotWidthInPixels)x\(screenshotHeightInPixels) pixels. \
        Locate the exact center of this UI element: "\(elementLabel)". \
        A previous rough estimate placed it near (\(Int(initialCoordinate.x)), \(Int(initialCoordinate.y))) — \
        verify that estimate and correct it if it is off. Respond with only the JSON object.
        """

        let requestBody: [String: Any] = [
            "model": model,
            "max_tokens": 128,
            "system": Self.locatorSystemPrompt,
            "messages": [
                [
                    "role": "user",
                    "content": [
                        [
                            "type": "image",
                            "source": [
                                "type": "base64",
                                "media_type": detectImageMediaType(for: screenshotData),
                                "data": screenshotData.base64EncodedString()
                            ]
                        ],
                        [
                            "type": "text",
                            "text": userPrompt
                        ]
                    ]
                ]
            ]
        ]

        var request = URLRequest(url: localBridgeChatURL)
        request.httpMethod = "POST"
        request.timeoutInterval = Self.locatorTimeoutSeconds
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")

        do {
            request.httpBody = try JSONSerialization.data(withJSONObject: requestBody)

            let (responseData, response) = try await session.data(for: request)

            guard let httpResponse = response as? HTTPURLResponse,
                  (200...299).contains(httpResponse.statusCode) else {
                let statusCode = (response as? HTTPURLResponse)?.statusCode ?? -1
                print("⚠️ ElementLocationDetector: locator request failed with status \(statusCode)")
                return nil
            }

            guard let responseText = extractAssistantText(fromAnthropicMessageData: responseData) else {
                print("⚠️ ElementLocationDetector: locator response had no text content")
                return nil
            }

            return parseRefinedCoordinate(
                fromLocatorResponseText: responseText,
                screenshotWidthInPixels: screenshotWidthInPixels,
                screenshotHeightInPixels: screenshotHeightInPixels
            )
        } catch {
            // Bridge not running, timeout, cancellation — the caller falls
            // back to the original tag coordinates, so just log and move on.
            print("⚠️ ElementLocationDetector: locator request failed: \(error.localizedDescription)")
            return nil
        }
    }

    // MARK: - Private Helpers

    /// Pulls the assistant's text out of an Anthropic-format message response
    /// (`{"content":[{"type":"text","text":"..."}]}`), which is what the
    /// bridge returns for non-streaming /chat requests.
    private func extractAssistantText(fromAnthropicMessageData responseData: Data) -> String? {
        guard let json = try? JSONSerialization.jsonObject(with: responseData) as? [String: Any],
              let contentBlocks = json["content"] as? [[String: Any]],
              let textBlock = contentBlocks.first(where: { ($0["type"] as? String) == "text" }),
              let text = textBlock["text"] as? String else {
            return nil
        }
        return text
    }

    /// Parses the locator's strict-JSON answer ({"x":123,"y":456,"found":true}).
    /// Tolerates surrounding prose or markdown fences by extracting the first
    /// {...} object from the text. Returns nil when found is false or the
    /// coordinate is outside the screenshot bounds.
    private func parseRefinedCoordinate(
        fromLocatorResponseText locatorResponseText: String,
        screenshotWidthInPixels: Int,
        screenshotHeightInPixels: Int
    ) -> CGPoint? {
        guard let jsonStartIndex = locatorResponseText.firstIndex(of: "{"),
              let jsonEndIndex = locatorResponseText.lastIndex(of: "}"),
              jsonStartIndex < jsonEndIndex else {
            print("⚠️ ElementLocationDetector: no JSON object in locator response: \(locatorResponseText.prefix(120))")
            return nil
        }

        let jsonSubstring = locatorResponseText[jsonStartIndex...jsonEndIndex]
        guard let jsonData = jsonSubstring.data(using: .utf8),
              let parsedJSON = try? JSONSerialization.jsonObject(with: jsonData) as? [String: Any] else {
            print("⚠️ ElementLocationDetector: locator JSON failed to parse: \(jsonSubstring.prefix(120))")
            return nil
        }

        guard let foundFlag = parsedJSON["found"] as? Bool, foundFlag,
              let xValue = parsedJSON["x"] as? NSNumber,
              let yValue = parsedJSON["y"] as? NSNumber else {
            print("🎯 ElementLocationDetector: element not found by locator")
            return nil
        }

        let refinedX = CGFloat(xValue.doubleValue)
        let refinedY = CGFloat(yValue.doubleValue)

        // A coordinate outside the screenshot means the model hallucinated —
        // the rough tag coordinate is safer than a clamped hallucination.
        guard refinedX >= 0, refinedX <= CGFloat(screenshotWidthInPixels),
              refinedY >= 0, refinedY <= CGFloat(screenshotHeightInPixels) else {
            print("⚠️ ElementLocationDetector: refined coordinate (\(Int(refinedX)), \(Int(refinedY))) is outside the \(screenshotWidthInPixels)x\(screenshotHeightInPixels) screenshot")
            return nil
        }

        print("🎯 ElementLocationDetector: refined coordinate (\(Int(refinedX)), \(Int(refinedY)))")
        return CGPoint(x: refinedX, y: refinedY)
    }

    /// Detects MIME type by inspecting the first bytes of image data.
    private func detectImageMediaType(for imageData: Data) -> String {
        if imageData.count >= 4 {
            let pngSignature: [UInt8] = [0x89, 0x50, 0x4E, 0x47]
            let firstFourBytes = [UInt8](imageData.prefix(4))
            if firstFourBytes == pngSignature {
                return "image/png"
            }
        }
        return "image/jpeg"
    }
}
