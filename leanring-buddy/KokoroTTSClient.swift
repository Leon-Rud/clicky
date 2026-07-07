//
//  KokoroTTSClient.swift
//  leanring-buddy
//
//  Speaks Claude's responses with the Kokoro-82M neural voice, running fully
//  on-device via the FluidAudio package (CoreML on the Apple Neural Engine).
//  Same speak/stop/isPlaying surface as AppleTTSClient so CompanionManager
//  can fall back to the system voice whenever Kokoro isn't ready.
//
//  Model files (~90 MB CoreML chain + G2P assets) are downloaded from
//  HuggingFace on first use and cached in ~/.cache/fluidaudio/Models/kokoro/.
//  Until that download finishes, speakText throws `modelNotReadyYet` so the
//  caller can fall back to Apple TTS instead of silently blocking for minutes.
//

import AVFoundation
import FluidAudio
import Foundation

enum KokoroTTSClientError: LocalizedError {
    /// The Kokoro CoreML models are still downloading or compiling. The
    /// caller should fall back to another TTS client for this utterance.
    case modelNotReadyYet

    var errorDescription: String? {
        switch self {
        case .modelNotReadyYet:
            return "Kokoro voice models are still downloading — not ready for this utterance yet."
        }
    }
}

@MainActor
final class KokoroTTSClient {

    /// FluidAudio's Kokoro synthesis actor. `af_heart` is the warm English
    /// voice that ships as the variant default; we pass it explicitly so a
    /// package-side default change can't silently switch voices.
    private let kokoroSynthesisManager = KokoroAneManager(
        variant: .english,
        defaultVoice: "af_heart"
    )

    /// Kokoro caps a single utterance at ~510 phonemes. English text maps to
    /// roughly one phoneme per character, so chunks capped well below that
    /// stay safely inside the limit while remaining long enough for natural
    /// prosody.
    private static let maximumCharactersPerSynthesisChunk = 280

    /// True once the CoreML chain + vocab + voice pack are loaded and
    /// synthesize calls will succeed without a long download stall.
    private(set) var isModelReady = false

    /// The in-flight model download/load task. Kept so repeated speak
    /// attempts don't start duplicate downloads; cleared on failure so the
    /// next attempt can retry (e.g. after the network comes back).
    private var modelPreparationTask: Task<Void, Never>?

    /// Player for the chunk currently sounding through the speakers.
    private var currentChunkAudioPlayer: AVAudioPlayer?

    /// Task that synthesizes and plays the remaining chunks of a multi-chunk
    /// utterance after speakText has already returned.
    private var remainingChunksPlaybackTask: Task<Void, Never>?

    /// True while a multi-chunk utterance still has chunks left to play, so
    /// isPlaying stays true in the brief gaps between chunk playbacks.
    private var hasRemainingSpeechChunks = false

    /// Incremented by stopPlayback / a new speakText so any in-flight chunk
    /// playback loop from a previous utterance knows to abandon its work.
    private var playbackGeneration = 0

    // MARK: - Model Preparation

    /// Kicks off the first-run model download + CoreML load in the
    /// background. Safe to call repeatedly; only one preparation runs at a
    /// time, and a failed preparation is cleared so a later call retries.
    func prepareModelsInBackground() {
        guard !isModelReady, modelPreparationTask == nil else { return }

        print("🔊 Kokoro TTS: preparing models (first run downloads ~90 MB to ~/.cache/fluidaudio/Models/kokoro/)")
        modelPreparationTask = Task {
            do {
                try await kokoroSynthesisManager.initialize()
                isModelReady = true
                print("🔊 Kokoro TTS: models ready")
            } catch {
                // Clear the task so a later speak attempt can retry.
                // CompanionManager already falls back to Apple TTS for any
                // utterance spoken while Kokoro is unavailable.
                modelPreparationTask = nil
                print("⚠️ Kokoro TTS: model preparation failed — \(error)")
            }
        }
    }

    // MARK: - Speaking

    /// Synthesizes `text` with Kokoro and starts playback. Returns once the
    /// first audio chunk is playing (matching AppleTTSClient's "returns when
    /// audio is playing" contract); longer utterances keep synthesizing and
    /// playing their remaining chunks in the background. Cancellation-safe.
    ///
    /// Throws `KokoroTTSClientError.modelNotReadyYet` if the first-run model
    /// download hasn't finished — callers should fall back to another voice
    /// for this utterance rather than waiting on the download.
    func speakText(_ text: String) async throws {
        try Task.checkCancellation()

        // Interrupt anything still speaking from a previous response.
        stopPlayback()

        guard isModelReady else {
            // Make sure the download is (still) running, then hand this
            // utterance back to the caller so it can use the fallback voice.
            prepareModelsInBackground()
            throw KokoroTTSClientError.modelNotReadyYet
        }

        let synthesisChunks = Self.splitTextIntoSynthesisChunks(
            text,
            maximumChunkLength: Self.maximumCharactersPerSynthesisChunk
        )
        guard let firstSynthesisChunk = synthesisChunks.first else { return }

        // Synthesize the first chunk inline so we only return once audio is
        // actually playing — the caller flips the UI to "responding" on
        // return. Output is a 24 kHz mono 16-bit PCM WAV.
        let firstChunkWavData = try await kokoroSynthesisManager.synthesize(text: firstSynthesisChunk)
        try Task.checkCancellation()

        playbackGeneration += 1
        let generationForThisUtterance = playbackGeneration

        let firstChunkPlayer = try AVAudioPlayer(data: firstChunkWavData)
        currentChunkAudioPlayer = firstChunkPlayer
        firstChunkPlayer.play()
        print("🔊 Kokoro TTS: speaking \(text.count) chars in \(synthesisChunks.count) chunk(s) with voice af_heart")

        let remainingSynthesisChunks = Array(synthesisChunks.dropFirst())
        guard !remainingSynthesisChunks.isEmpty else { return }

        hasRemainingSpeechChunks = true
        remainingChunksPlaybackTask = Task {
            await playRemainingChunks(remainingSynthesisChunks, generation: generationForThisUtterance)
        }
    }

    /// Whether TTS audio is currently playing back (or a multi-chunk
    /// utterance still has chunks queued to play).
    var isPlaying: Bool {
        (currentChunkAudioPlayer?.isPlaying ?? false) || hasRemainingSpeechChunks
    }

    /// Stops any in-progress playback immediately, including queued chunks.
    func stopPlayback() {
        playbackGeneration += 1
        remainingChunksPlaybackTask?.cancel()
        remainingChunksPlaybackTask = nil
        hasRemainingSpeechChunks = false
        currentChunkAudioPlayer?.stop()
        currentChunkAudioPlayer = nil
    }

    // MARK: - Multi-Chunk Playback

    /// Sequentially synthesizes and plays `remainingSynthesisChunks`. Each
    /// chunk is synthesized while the previous one is still sounding, so
    /// there is no synthesis stall between chunks. Abandons its work as soon
    /// as the playback generation changes (stopPlayback or a new utterance).
    private func playRemainingChunks(_ remainingSynthesisChunks: [String], generation: Int) async {
        for synthesisChunk in remainingSynthesisChunks {
            let chunkWavData: Data
            do {
                chunkWavData = try await kokoroSynthesisManager.synthesize(text: synthesisChunk)
            } catch {
                // A mid-utterance synthesis failure truncates the response
                // rather than switching voices halfway through a reply.
                print("⚠️ Kokoro TTS: chunk synthesis failed mid-utterance — \(error)")
                break
            }

            // Wait for the currently sounding chunk to finish.
            while playbackGeneration == generation,
                  let soundingPlayer = currentChunkAudioPlayer,
                  soundingPlayer.isPlaying {
                try? await Task.sleep(nanoseconds: 50_000_000)
                if Task.isCancelled { return }
            }
            guard playbackGeneration == generation, !Task.isCancelled else { return }

            guard let nextChunkPlayer = try? AVAudioPlayer(data: chunkWavData) else { break }
            currentChunkAudioPlayer = nextChunkPlayer
            nextChunkPlayer.play()
        }

        // Only clear the flag if this loop still owns playback — a newer
        // utterance may have taken over while we were synthesizing.
        if playbackGeneration == generation {
            hasRemainingSpeechChunks = false
        }
    }

    // MARK: - Text Chunking

    /// Splits `text` into sentence-aligned chunks no longer than
    /// `maximumChunkLength` characters, so each chunk stays under Kokoro's
    /// per-utterance phoneme cap. Sentences are kept whole where possible;
    /// a single overlong sentence is split at word boundaries.
    static func splitTextIntoSynthesisChunks(_ text: String, maximumChunkLength: Int) -> [String] {
        let trimmedText = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmedText.isEmpty else { return [] }
        guard trimmedText.count > maximumChunkLength else { return [trimmedText] }

        // Break the text into sentences, keeping each terminator with its
        // sentence so the synthesized prosody sounds natural.
        var sentences: [String] = []
        var sentenceAccumulator = ""
        for character in trimmedText {
            sentenceAccumulator.append(character)
            if character == "." || character == "!" || character == "?" || character == "\n" {
                let trimmedSentence = sentenceAccumulator.trimmingCharacters(in: .whitespacesAndNewlines)
                if !trimmedSentence.isEmpty {
                    sentences.append(trimmedSentence)
                }
                sentenceAccumulator = ""
            }
        }
        let trailingSentence = sentenceAccumulator.trimmingCharacters(in: .whitespacesAndNewlines)
        if !trailingSentence.isEmpty {
            sentences.append(trailingSentence)
        }

        // Split any single sentence that is itself over the limit at word
        // boundaries so no chunk can exceed the phoneme cap.
        var lengthLimitedSentences: [String] = []
        for sentence in sentences {
            if sentence.count <= maximumChunkLength {
                lengthLimitedSentences.append(sentence)
                continue
            }
            var wordAccumulator = ""
            for word in sentence.split(separator: " ") {
                if wordAccumulator.isEmpty {
                    wordAccumulator = String(word)
                } else if wordAccumulator.count + 1 + word.count <= maximumChunkLength {
                    wordAccumulator += " " + word
                } else {
                    lengthLimitedSentences.append(wordAccumulator)
                    wordAccumulator = String(word)
                }
            }
            if !wordAccumulator.isEmpty {
                lengthLimitedSentences.append(wordAccumulator)
            }
        }

        // Greedily pack consecutive sentences into chunks up to the limit so
        // short sentences don't each pay a separate synthesis round-trip.
        var synthesisChunks: [String] = []
        var chunkAccumulator = ""
        for sentence in lengthLimitedSentences {
            if chunkAccumulator.isEmpty {
                chunkAccumulator = sentence
            } else if chunkAccumulator.count + 1 + sentence.count <= maximumChunkLength {
                chunkAccumulator += " " + sentence
            } else {
                synthesisChunks.append(chunkAccumulator)
                chunkAccumulator = sentence
            }
        }
        if !chunkAccumulator.isEmpty {
            synthesisChunks.append(chunkAccumulator)
        }
        return synthesisChunks
    }
}
