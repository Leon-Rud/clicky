//
//  AppleSpeechAnalyzerTranscriptionProvider.swift
//  leanring-buddy
//
//  On-device streaming transcription backed by Apple's SpeechAnalyzer +
//  SpeechTranscriber API (macOS 26 "Tahoe" and later). Replaces the
//  AssemblyAI websocket provider so no audio ever leaves the machine.
//

import AVFoundation
import Foundation
import Speech

struct AppleSpeechAnalyzerTranscriptionProviderError: LocalizedError {
    let message: String

    var errorDescription: String? {
        message
    }
}

@available(macOS 26.0, *)
final class AppleSpeechAnalyzerTranscriptionProvider: BuddyTranscriptionProvider {
    let displayName = "Apple Speech (on-device)"
    // The SpeechAnalyzer/SpeechTranscriber API performs on-device analysis and
    // does not go through SFSpeechRecognizer's authorization flow — only the
    // microphone permission (which the dictation manager already requests) is
    // needed.
    // FORK-TODO: verify on first build that no SFSpeechRecognizer authorization
    // prompt is required; if transcription silently fails, flip this to true.
    let requiresSpeechRecognitionPermission = false

    var isConfigured: Bool { true }
    var unavailableExplanation: String? { nil }

    func startStreamingSession(
        keyterms: [String],
        onTranscriptUpdate: @escaping (String) -> Void,
        onFinalTranscriptReady: @escaping (String) -> Void,
        onError: @escaping (Error) -> Void
    ) async throws -> any BuddyStreamingTranscriptionSession {
        // Note: SpeechTranscriber has no keyterm-boosting equivalent to
        // AssemblyAI's keyterms_prompt, so `keyterms` is intentionally unused.
        let locale = try await Self.resolveSupportedLocale()

        let transcriber = SpeechTranscriber(
            locale: locale,
            transcriptionOptions: [],
            reportingOptions: [.volatileResults],
            attributeOptions: []
        )

        // Download the on-device speech model for this locale if it isn't
        // installed yet. This is a no-op once the assets are present.
        // FORK-TODO: confirm the AssetInventory API shape compiles as written —
        // `assetInstallationRequest(supporting:)` returning an optional request
        // with `downloadAndInstall()` is the documented macOS 26 surface.
        if let installationRequest = try await AssetInventory.assetInstallationRequest(supporting: [transcriber]) {
            print("🎙️ Apple SpeechAnalyzer: downloading on-device model for \(locale.identifier)…")
            try await installationRequest.downloadAndInstall()
        }

        let session = try await AppleSpeechAnalyzerTranscriptionSession(
            transcriber: transcriber,
            onTranscriptUpdate: onTranscriptUpdate,
            onFinalTranscriptReady: onFinalTranscriptReady,
            onError: onError
        )

        return session
    }

    /// Picks the user's locale if SpeechTranscriber supports it, falling back
    /// to en-US, and errors out if neither is supported on this machine.
    private static func resolveSupportedLocale() async throws -> Locale {
        let supportedLocales = await SpeechTranscriber.supportedLocales

        let preferredLocales = [
            Locale.autoupdatingCurrent,
            Locale(identifier: "en-US")
        ]

        for preferredLocale in preferredLocales {
            if supportedLocales.contains(where: {
                $0.identifier(.bcp47) == preferredLocale.identifier(.bcp47)
            }) {
                return preferredLocale
            }
        }

        if let firstSupportedLocale = supportedLocales.first {
            return firstSupportedLocale
        }

        throw AppleSpeechAnalyzerTranscriptionProviderError(
            message: "on-device dictation isn't available on this mac."
        )
    }
}

@available(macOS 26.0, *)
private final class AppleSpeechAnalyzerTranscriptionSession: BuddyStreamingTranscriptionSession {
    let finalTranscriptFallbackDelaySeconds: TimeInterval = 1.5

    private let transcriber: SpeechTranscriber
    private let analyzer: SpeechAnalyzer
    private let inputBuilder: AsyncStream<AnalyzerInput>.Continuation
    private let analyzerAudioFormat: AVAudioFormat?

    private let onTranscriptUpdate: (String) -> Void
    private let onFinalTranscriptReady: (String) -> Void
    private let onError: (Error) -> Void

    /// Serializes access to transcript state — results arrive on the
    /// recognition task while append/finish arrive from the audio tap and
    /// main thread.
    private let stateQueue = DispatchQueue(label: "com.learningbuddy.applespeechanalyzer.state")

    /// Converts mic-format buffers (typically Float32 @ 44.1/48kHz) into the
    /// analyzer's preferred format. Only touched from the audio tap thread.
    private var audioConverter: AVAudioConverter?
    private var currentInputFormatDescription: String?

    private var finalizedTranscriptSegments: [String] = []
    private var volatileTranscriptText = ""
    private var hasRequestedFinalTranscript = false
    private var hasDeliveredFinalTranscript = false

    private var recognitionTask: Task<Void, Never>?

    init(
        transcriber: SpeechTranscriber,
        onTranscriptUpdate: @escaping (String) -> Void,
        onFinalTranscriptReady: @escaping (String) -> Void,
        onError: @escaping (Error) -> Void
    ) async throws {
        self.transcriber = transcriber
        self.onTranscriptUpdate = onTranscriptUpdate
        self.onFinalTranscriptReady = onFinalTranscriptReady
        self.onError = onError

        self.analyzer = SpeechAnalyzer(modules: [transcriber])

        // The analyzer publishes the audio format it wants input in.
        // FORK-TODO: `bestAvailableAudioFormat(compatibleWith:)` is the
        // documented static helper; if the compiler flags it, check
        // `SpeechAnalyzer.bestAvailableAudioFormat(compatibleWith:considering:)`.
        self.analyzerAudioFormat = await SpeechAnalyzer.bestAvailableAudioFormat(compatibleWith: [transcriber])

        let (inputSequence, inputBuilder) = AsyncStream<AnalyzerInput>.makeStream()
        self.inputBuilder = inputBuilder

        try await analyzer.start(inputSequence: inputSequence)

        startRecognitionTask()
    }

    // MARK: - BuddyStreamingTranscriptionSession

    func appendAudioBuffer(_ audioBuffer: AVAudioPCMBuffer) {
        var shouldAppend = false
        stateQueue.sync {
            shouldAppend = !hasRequestedFinalTranscript
        }
        guard shouldAppend else { return }

        guard let convertedBuffer = convertToAnalyzerFormat(audioBuffer) else { return }
        inputBuilder.yield(AnalyzerInput(buffer: convertedBuffer))
    }

    func requestFinalTranscript() {
        var shouldFinalize = false
        stateQueue.sync {
            if !hasRequestedFinalTranscript {
                hasRequestedFinalTranscript = true
                shouldFinalize = true
            }
        }
        guard shouldFinalize else { return }

        // Close the input stream, then ask the analyzer to flush any pending
        // audio through the transcriber. The results stream ends after the
        // last (final) result, which is where the final transcript is delivered.
        inputBuilder.finish()
        Task { [analyzer] in
            do {
                try await analyzer.finalizeAndFinishThroughEndOfInput()
            } catch {
                // Delivery still happens via the fallback path below — the
                // recognition task ends (or errors) and hands over whatever
                // transcript text has accumulated.
                print("[AppleSpeechAnalyzer] ⚠️ finalize failed: \(error.localizedDescription)")
            }
        }
    }

    func cancel() {
        inputBuilder.finish()
        recognitionTask?.cancel()
        recognitionTask = nil
        Task { [analyzer] in
            await analyzer.cancelAndFinishNow()
        }
    }

    // MARK: - Recognition results

    private func startRecognitionTask() {
        recognitionTask = Task { [weak self] in
            guard let self else { return }
            do {
                for try await result in self.transcriber.results {
                    self.handleTranscriberResult(result)
                }
                self.handleResultsStreamEnded()
            } catch {
                self.handleRecognitionFailure(error)
            }
        }
    }

    private func handleTranscriberResult(_ result: SpeechTranscriber.Result) {
        let resultText = String(result.text.characters)
            .trimmingCharacters(in: .whitespacesAndNewlines)

        stateQueue.async {
            if result.isFinal {
                // A finalized result replaces the volatile text that previewed it.
                self.volatileTranscriptText = ""
                if !resultText.isEmpty {
                    self.finalizedTranscriptSegments.append(resultText)
                }
            } else {
                self.volatileTranscriptText = resultText
            }

            let fullTranscriptText = self.composeFullTranscript()
            if !fullTranscriptText.isEmpty {
                self.onTranscriptUpdate(fullTranscriptText)
            }
        }
    }

    /// Called when the transcriber's results stream completes — i.e. the
    /// analyzer finished processing all input after `requestFinalTranscript`.
    private func handleResultsStreamEnded() {
        stateQueue.async {
            self.deliverFinalTranscriptIfNeeded(self.composeFullTranscript())
        }
    }

    private func handleRecognitionFailure(_ error: Error) {
        stateQueue.async {
            let latestTranscriptText = self.composeFullTranscript()

            // Mirror the AssemblyAI behavior: if the user already released the
            // hotkey and we have text, deliver it instead of surfacing an error.
            if self.hasRequestedFinalTranscript && !latestTranscriptText.isEmpty {
                print("[AppleSpeechAnalyzer] ⚠️ error during finalization, delivering partial transcript: \(error.localizedDescription)")
                self.deliverFinalTranscriptIfNeeded(latestTranscriptText)
                return
            }

            print("[AppleSpeechAnalyzer] ❌ session failed: \(error.localizedDescription)")
            self.onError(error)
        }
    }

    /// Must be called on stateQueue.
    private func deliverFinalTranscriptIfNeeded(_ transcriptText: String) {
        guard !hasDeliveredFinalTranscript else { return }
        hasDeliveredFinalTranscript = true
        onFinalTranscriptReady(transcriptText)
    }

    /// Must be called on stateQueue.
    private func composeFullTranscript() -> String {
        var transcriptSegments = finalizedTranscriptSegments

        let trimmedVolatileText = volatileTranscriptText
            .trimmingCharacters(in: .whitespacesAndNewlines)
        if !trimmedVolatileText.isEmpty {
            transcriptSegments.append(trimmedVolatileText)
        }

        return transcriptSegments
            .joined(separator: " ")
            .trimmingCharacters(in: .whitespacesAndNewlines)
    }

    // MARK: - Audio conversion

    /// Converts a mic buffer into the analyzer's preferred format. Returns the
    /// buffer unchanged when no conversion is needed (or no format was published).
    private func convertToAnalyzerFormat(_ audioBuffer: AVAudioPCMBuffer) -> AVAudioPCMBuffer? {
        guard let analyzerAudioFormat else { return audioBuffer }
        guard audioBuffer.format != analyzerAudioFormat else { return audioBuffer }

        let inputFormatDescription = audioBuffer.format.settings.description
        if currentInputFormatDescription != inputFormatDescription {
            audioConverter = AVAudioConverter(from: audioBuffer.format, to: analyzerAudioFormat)
            currentInputFormatDescription = inputFormatDescription
        }

        guard let audioConverter else { return nil }

        let sampleRateRatio = analyzerAudioFormat.sampleRate / audioBuffer.format.sampleRate
        let outputFrameCapacity = AVAudioFrameCount(
            (Double(audioBuffer.frameLength) * sampleRateRatio).rounded(.up) + 32
        )

        guard let outputBuffer = AVAudioPCMBuffer(
            pcmFormat: analyzerAudioFormat,
            frameCapacity: outputFrameCapacity
        ) else {
            return nil
        }

        var hasProvidedSourceBuffer = false
        var conversionError: NSError?

        let conversionStatus = audioConverter.convert(to: outputBuffer, error: &conversionError) { _, outStatus in
            if hasProvidedSourceBuffer {
                outStatus.pointee = .noDataNow
                return nil
            }

            hasProvidedSourceBuffer = true
            outStatus.pointee = .haveData
            return audioBuffer
        }

        guard conversionStatus != .error, outputBuffer.frameLength > 0 else { return nil }
        return outputBuffer
    }

    deinit {
        recognitionTask?.cancel()
        inputBuilder.finish()
    }
}
