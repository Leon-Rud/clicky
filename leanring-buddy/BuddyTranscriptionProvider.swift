//
//  BuddyTranscriptionProvider.swift
//  leanring-buddy
//
//  Shared protocol surface for voice transcription backends.
//

import AVFoundation
import Foundation

protocol BuddyStreamingTranscriptionSession: AnyObject {
    var finalTranscriptFallbackDelaySeconds: TimeInterval { get }
    func appendAudioBuffer(_ audioBuffer: AVAudioPCMBuffer)
    func requestFinalTranscript()
    func cancel()
}

protocol BuddyTranscriptionProvider {
    var displayName: String { get }
    var requiresSpeechRecognitionPermission: Bool { get }
    var isConfigured: Bool { get }
    var unavailableExplanation: String? { get }

    func startStreamingSession(
        keyterms: [String],
        onTranscriptUpdate: @escaping (String) -> Void,
        onFinalTranscriptReady: @escaping (String) -> Void,
        onError: @escaping (Error) -> Void
    ) async throws -> any BuddyStreamingTranscriptionSession
}

enum BuddyTranscriptionProviderFactory {
    static func makeDefaultProvider() -> any BuddyTranscriptionProvider {
        let provider = resolveProvider()
        print("🎙️ Transcription: using \(provider.displayName)")
        return provider
    }

    /// Fork: transcription is fully on-device. The cloud providers
    /// (AssemblyAI, OpenAI) are kept in the tree but never constructed.
    /// macOS 26+ uses the SpeechAnalyzer/SpeechTranscriber API; older
    /// systems fall back to the legacy SFSpeechRecognizer provider.
    private static func resolveProvider() -> any BuddyTranscriptionProvider {
        if #available(macOS 26.0, *) {
            return AppleSpeechAnalyzerTranscriptionProvider()
        }

        return AppleSpeechTranscriptionProvider()
    }
}
