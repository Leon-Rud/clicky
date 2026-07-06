//
//  AppleTTSClient.swift
//  leanring-buddy
//
//  Speaks Claude's responses using Apple's on-device AVSpeechSynthesizer.
//  Drop-in replacement for the ElevenLabs client — same speak/stop/isPlaying
//  surface — with zero network calls and zero cost.
//

import AVFoundation
import Foundation

@MainActor
final class AppleTTSClient {
    /// Kept alive for the lifetime of the client — creating a fresh
    /// synthesizer per utterance can cut off playback when the old one
    /// deallocates.
    private let speechSynthesizer = AVSpeechSynthesizer()

    /// The best system voice for the user's language, resolved once.
    /// Prefers premium, then enhanced, then default quality.
    private lazy var preferredVoice: AVSpeechSynthesisVoice? = Self.resolvePreferredVoice()

    /// Speaks `text` through the system speech synthesizer. Returns once
    /// playback has been enqueued (which is effectively when it starts),
    /// matching the previous TTS client's "returns when audio is playing"
    /// contract. Cancellation-safe.
    func speakText(_ text: String) async throws {
        try Task.checkCancellation()

        // Interrupt anything still speaking from a previous response.
        if speechSynthesizer.isSpeaking {
            speechSynthesizer.stopSpeaking(at: .immediate)
        }

        let utterance = AVSpeechUtterance(string: text)
        if let preferredVoice {
            utterance.voice = preferredVoice
        }
        // Slightly above the default rate reads more naturally for short
        // conversational replies.
        utterance.rate = AVSpeechUtteranceDefaultSpeechRate * 1.05
        utterance.prefersAssistiveTechnologySettings = false

        speechSynthesizer.speak(utterance)
        print("🔊 Apple TTS: speaking \(text.count) chars with voice \(preferredVoice?.name ?? "system default")")
    }

    /// Whether TTS audio is currently playing back.
    var isPlaying: Bool {
        speechSynthesizer.isSpeaking
    }

    /// Stops any in-progress playback immediately.
    func stopPlayback() {
        speechSynthesizer.stopSpeaking(at: .immediate)
    }

    /// Picks the highest-quality installed system voice for the user's
    /// language: premium > enhanced > default. Users can install premium
    /// voices in System Settings > Accessibility > Spoken Content.
    private static func resolvePreferredVoice() -> AVSpeechSynthesisVoice? {
        let preferredLanguageCodes = [
            AVSpeechSynthesisVoice.currentLanguageCode(),
            "en-US"
        ]

        let allVoices = AVSpeechSynthesisVoice.speechVoices()

        for languageCode in preferredLanguageCodes {
            let languagePrefix = String(languageCode.prefix(2))
            let matchingVoices = allVoices.filter { $0.language.hasPrefix(languagePrefix) }
            guard !matchingVoices.isEmpty else { continue }

            if let premiumVoice = matchingVoices.first(where: { $0.quality == .premium }) {
                return premiumVoice
            }
            if let enhancedVoice = matchingVoices.first(where: { $0.quality == .enhanced }) {
                return enhancedVoice
            }
            return AVSpeechSynthesisVoice(language: languageCode) ?? matchingVoices.first
        }

        return nil
    }
}
