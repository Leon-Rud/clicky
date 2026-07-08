//
//  CompanionManager.swift
//  leanring-buddy
//
//  Central state manager for the companion voice mode. Owns the push-to-talk
//  pipeline (dictation manager + global shortcut monitor + overlay) and
//  exposes observable voice state for the panel UI.
//

import AVFoundation
import Combine
import Foundation
import ScreenCaptureKit
import SwiftUI

enum CompanionVoiceState {
    case idle
    case listening
    case processing
    case responding
}

@MainActor
final class CompanionManager: ObservableObject {
    @Published private(set) var voiceState: CompanionVoiceState = .idle
    @Published private(set) var lastTranscript: String?
    @Published private(set) var currentAudioPowerLevel: CGFloat = 0
    @Published private(set) var hasAccessibilityPermission = false
    @Published private(set) var hasScreenRecordingPermission = false
    @Published private(set) var hasMicrophonePermission = false
    @Published private(set) var hasScreenContentPermission = false

    /// Screen location (global AppKit coords) of a detected UI element the
    /// buddy should fly to and point at. Parsed from Claude's response;
    /// observed by BlueCursorView to trigger the flight animation.
    @Published var detectedElementScreenLocation: CGPoint?
    /// The display frame (global AppKit coords) of the screen the detected
    /// element is on, so BlueCursorView knows which screen overlay should animate.
    @Published var detectedElementDisplayFrame: CGRect?
    /// Custom speech bubble text for the pointing animation. When set,
    /// BlueCursorView uses this instead of a random pointer phrase.
    @Published var detectedElementBubbleText: String?

    // MARK: - Onboarding Video State (shared across all screen overlays)

    @Published var onboardingVideoPlayer: AVPlayer?
    @Published var showOnboardingVideo: Bool = false
    @Published var onboardingVideoOpacity: Double = 0.0
    private var onboardingVideoEndObserver: NSObjectProtocol?
    private var onboardingDemoTimeObserver: Any?

    // MARK: - Onboarding Prompt Bubble

    /// Text streamed character-by-character on the cursor after the onboarding video ends.
    @Published var onboardingPromptText: String = ""
    @Published var onboardingPromptOpacity: Double = 0.0
    @Published var showOnboardingPrompt: Bool = false

    // MARK: - Onboarding Music

    private var onboardingMusicPlayer: AVAudioPlayer?
    private var onboardingMusicFadeTimer: Timer?

    let buddyDictationManager = BuddyDictationManager()
    let globalPushToTalkShortcutMonitor = GlobalPushToTalkShortcutMonitor()
    let overlayWindowManager = OverlayWindowManager()
    // Response text is now displayed inline on the cursor overlay via
    // streamingResponseText, so no separate response overlay manager is needed.

    private lazy var claudeAPI: ClaudeAPI = {
        return ClaudeAPI(model: selectedModel)
    }()

    /// Precision second pass for pointing/click coordinates. Given the same
    /// screenshot sent with the main request and the element label from a
    /// [POINT:...]/[CLICK:...] tag, it asks Claude (via the local bridge) for
    /// refined exact coordinates. Falls back to the tag coordinates on failure.
    private lazy var elementLocationDetector: ElementLocationDetector = {
        return ElementLocationDetector()
    }()

    /// On-device text-to-speech via AVSpeechSynthesizer. Replaces the
    /// ElevenLabs client — no network, no API key.
    private lazy var appleTTSClient: AppleTTSClient = {
        return AppleTTSClient()
    }()

    /// On-device neural text-to-speech via the Kokoro-82M model (FluidAudio
    /// package, CoreML on the Neural Engine). Preferred voice when the
    /// "PreferKokoroVoice" setting is on; appleTTSClient is the fallback
    /// whenever Kokoro isn't ready or fails.
    private lazy var kokoroTTSClient: KokoroTTSClient = {
        return KokoroTTSClient()
    }()

    /// Conversation history so Claude remembers prior exchanges within a session.
    /// Each entry is the user's transcript and Claude's response.
    private var conversationHistory: [(userTranscript: String, assistantResponse: String)] = []

    /// The currently running AI response task, if any. Cancelled when the user
    /// speaks again so a new response can begin immediately.
    private var currentResponseTask: Task<Void, Never>?

    private var shortcutTransitionCancellable: AnyCancellable?
    private var voiceStateCancellable: AnyCancellable?
    private var audioPowerCancellable: AnyCancellable?
    private var accessibilityCheckTimer: Timer?
    private var pendingKeyboardShortcutStartTask: Task<Void, Never>?
    /// Scheduled hide for transient cursor mode — cancelled if the user
    /// speaks again before the delay elapses.
    private var transientHideTask: Task<Void, Never>?

    /// True when all three required permissions (accessibility, screen recording,
    /// microphone) are granted. Used by the panel to show a single "all good" state.
    var allPermissionsGranted: Bool {
        hasAccessibilityPermission && hasScreenRecordingPermission && hasMicrophonePermission && hasScreenContentPermission
    }

    /// Whether the blue cursor overlay is currently visible on screen.
    /// Used by the panel to show accurate status text ("Active" vs "Ready").
    @Published private(set) var isOverlayVisible: Bool = false

    /// The Claude model used for voice responses. Persisted to UserDefaults.
    @Published var selectedModel: String = UserDefaults.standard.string(forKey: "selectedClaudeModel") ?? "claude-sonnet-4-6"

    func setSelectedModel(_ model: String) {
        selectedModel = model
        UserDefaults.standard.set(model, forKey: "selectedClaudeModel")
        claudeAPI.model = model
    }

    /// The user's Anthropic API key. Entered in the panel, persisted to
    /// UserDefaults ("AnthropicAPIKey"), and read by ClaudeAPI on each request.
    @Published var anthropicAPIKey: String = AnthropicAPIKeyStore.apiKey ?? ""

    /// Whether an API key has been entered — the app can't respond without one.
    var hasAnthropicAPIKey: Bool {
        !anthropicAPIKey.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
    }

    func setAnthropicAPIKey(_ apiKey: String) {
        anthropicAPIKey = apiKey
        AnthropicAPIKeyStore.setAPIKey(apiKey)
    }

    /// How the app reaches Claude: the local subscription bridge (default) or
    /// a direct Anthropic API key. Persisted to UserDefaults ("ClaudeBackendMode")
    /// and read by ClaudeAPI on each request.
    @Published var claudeBackendMode: ClaudeBackendMode = ClaudeBackendModeStore.mode

    func setClaudeBackendMode(_ claudeBackendMode: ClaudeBackendMode) {
        self.claudeBackendMode = claudeBackendMode
        ClaudeBackendModeStore.setMode(claudeBackendMode)
    }

    /// User preference for whether the Clicky cursor should be shown.
    /// When toggled off, the overlay is hidden and push-to-talk is disabled.
    /// Persisted to UserDefaults so the choice survives app restarts.
    @Published var isClickyCursorEnabled: Bool = UserDefaults.standard.object(forKey: "isClickyCursorEnabled") == nil
        ? true
        : UserDefaults.standard.bool(forKey: "isClickyCursorEnabled")

    func setClickyCursorEnabled(_ enabled: Bool) {
        isClickyCursorEnabled = enabled
        UserDefaults.standard.set(enabled, forKey: "isClickyCursorEnabled")
        transientHideTask?.cancel()
        transientHideTask = nil

        if enabled {
            overlayWindowManager.hasShownOverlayBefore = true
            overlayWindowManager.showOverlay(onScreens: NSScreen.screens, companionManager: self)
            isOverlayVisible = true
        } else {
            overlayWindowManager.hideOverlay()
            isOverlayVisible = false
        }
    }

    /// User preference for whether Clicky may perform real mouse clicks when
    /// the user verbally asks for an action ("click the save button"). When
    /// off, [CLICK:...] tags from Claude degrade to pointing-only behavior.
    /// Persisted to UserDefaults ("AllowClickActions"), default true.
    @Published var allowClickActions: Bool = UserDefaults.standard.object(forKey: "AllowClickActions") == nil
        ? true
        : UserDefaults.standard.bool(forKey: "AllowClickActions")

    func setAllowClickActions(_ allowClickActions: Bool) {
        self.allowClickActions = allowClickActions
        UserDefaults.standard.set(allowClickActions, forKey: "AllowClickActions")
    }

    /// Whether POINT coordinates should also be refined through the locator
    /// second pass ("PrecisePointing" in UserDefaults, default true). CLICK
    /// coordinates are ALWAYS refined regardless of this flag — accuracy
    /// matters more when performing a real click than when pointing.
    private var isPrecisePointingEnabled: Bool {
        UserDefaults.standard.object(forKey: "PrecisePointing") == nil
            ? true
            : UserDefaults.standard.bool(forKey: "PrecisePointing")
    }

    /// User preference for the response voice: Kokoro's on-device neural
    /// voice (default) or the Apple system voice. Persisted to UserDefaults
    /// ("PreferKokoroVoice"). When Kokoro can't speak an utterance (models
    /// still downloading, synthesis error), the app falls back to the Apple
    /// system voice for that utterance instead of staying silent.
    @Published var preferKokoroVoice: Bool = UserDefaults.standard.object(forKey: "PreferKokoroVoice") == nil
        ? true
        : UserDefaults.standard.bool(forKey: "PreferKokoroVoice")

    func setPreferKokoroVoice(_ preferKokoroVoice: Bool) {
        self.preferKokoroVoice = preferKokoroVoice
        UserDefaults.standard.set(preferKokoroVoice, forKey: "PreferKokoroVoice")
        if preferKokoroVoice {
            // Warm the Kokoro models right away so switching voices doesn't
            // leave the next few utterances on the fallback voice while the
            // first-run download/compile finishes.
            kokoroTTSClient.prepareModelsInBackground()
        }
    }

    /// Whether the user has completed onboarding at least once. Persisted
    /// to UserDefaults so the Start button only appears on first launch.
    var hasCompletedOnboarding: Bool {
        get { UserDefaults.standard.bool(forKey: "hasCompletedOnboarding") }
        set { UserDefaults.standard.set(newValue, forKey: "hasCompletedOnboarding") }
    }

    /// Whether the user has submitted their email during onboarding.
    @Published var hasSubmittedEmail: Bool = UserDefaults.standard.bool(forKey: "hasSubmittedEmail")

    /// Submits the user's email to FormSpark.
    func submitEmail(_ email: String) {
        let trimmedEmail = email.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmedEmail.isEmpty else { return }

        hasSubmittedEmail = true
        UserDefaults.standard.set(true, forKey: "hasSubmittedEmail")

        // Submit to FormSpark
        Task {
            var request = URLRequest(url: URL(string: "https://submit-form.com/RWbGJxmIs")!)
            request.httpMethod = "POST"
            request.setValue("application/json", forHTTPHeaderField: "Content-Type")
            request.httpBody = try? JSONSerialization.data(withJSONObject: ["email": trimmedEmail])
            _ = try? await URLSession.shared.data(for: request)
        }
    }

    func start() {
        refreshAllPermissions()
        print("🔑 Clicky start — accessibility: \(hasAccessibilityPermission), screen: \(hasScreenRecordingPermission), mic: \(hasMicrophonePermission), screenContent: \(hasScreenContentPermission), onboarded: \(hasCompletedOnboarding)")
        startPermissionPolling()
        bindVoiceStateObservation()
        bindAudioPowerLevel()
        bindShortcutTransitions()
        // Eagerly touch the Claude API so its TLS warmup handshake completes
        // well before the onboarding demo fires at ~40s into the video.
        _ = claudeAPI

        // Warm the Kokoro voice models at launch (first run downloads ~90 MB
        // in the background) so the first spoken response can use the neural
        // voice. Utterances spoken before the models are ready fall back to
        // the Apple system voice.
        if preferKokoroVoice {
            kokoroTTSClient.prepareModelsInBackground()
        }

        // If the user already completed onboarding AND all permissions are
        // still granted, show the cursor overlay immediately. If permissions
        // were revoked (e.g. signing change), don't show the cursor — the
        // panel will show the permissions UI instead.
        if hasCompletedOnboarding && allPermissionsGranted && isClickyCursorEnabled {
            overlayWindowManager.hasShownOverlayBefore = true
            overlayWindowManager.showOverlay(onScreens: NSScreen.screens, companionManager: self)
            isOverlayVisible = true
        }
    }

    /// Called by BlueCursorView after the buddy finishes its pointing
    /// animation and returns to cursor-following mode.
    /// Triggers the onboarding sequence — dismisses the panel and restarts
    /// the overlay so the welcome animation and intro video play.
    func triggerOnboarding() {
        // Post notification so the panel manager can dismiss the panel
        NotificationCenter.default.post(name: .clickyDismissPanel, object: nil)

        // Mark onboarding as completed so the Start button won't appear
        // again on future launches — the cursor will auto-show instead
        hasCompletedOnboarding = true

        ClickyAnalytics.trackOnboardingStarted()

        // Play Besaid theme at 60% volume, fade out after 1m 30s
        startOnboardingMusic()

        // Show the overlay for the first time — isFirstAppearance triggers
        // the welcome animation and onboarding video
        overlayWindowManager.showOverlay(onScreens: NSScreen.screens, companionManager: self)
        isOverlayVisible = true
    }

    /// Replays the onboarding experience from the "Watch Onboarding Again"
    /// footer link. Same flow as triggerOnboarding but the cursor overlay
    /// is already visible so we just restart the welcome animation and video.
    func replayOnboarding() {
        NotificationCenter.default.post(name: .clickyDismissPanel, object: nil)
        ClickyAnalytics.trackOnboardingReplayed()
        startOnboardingMusic()
        // Tear down any existing overlays and recreate with isFirstAppearance = true
        overlayWindowManager.hasShownOverlayBefore = false
        overlayWindowManager.showOverlay(onScreens: NSScreen.screens, companionManager: self)
        isOverlayVisible = true
    }

    private func stopOnboardingMusic() {
        onboardingMusicFadeTimer?.invalidate()
        onboardingMusicFadeTimer = nil
        onboardingMusicPlayer?.stop()
        onboardingMusicPlayer = nil
    }

    private func startOnboardingMusic() {
        stopOnboardingMusic()
        guard let musicURL = Bundle.main.url(forResource: "ff", withExtension: "mp3") else {
            print("⚠️ Clicky: ff.mp3 not found in bundle")
            return
        }

        do {
            let player = try AVAudioPlayer(contentsOf: musicURL)
            player.volume = 0.3
            player.play()
            self.onboardingMusicPlayer = player

            // After 1m 30s, fade the music out over 3s
            onboardingMusicFadeTimer = Timer.scheduledTimer(withTimeInterval: 90.0, repeats: false) { [weak self] _ in
                self?.fadeOutOnboardingMusic()
            }
        } catch {
            print("⚠️ Clicky: Failed to play onboarding music: \(error)")
        }
    }

    private func fadeOutOnboardingMusic() {
        guard let player = onboardingMusicPlayer else { return }

        let fadeSteps = 30
        let fadeDuration: Double = 3.0
        let stepInterval = fadeDuration / Double(fadeSteps)
        let volumeDecrement = player.volume / Float(fadeSteps)
        var stepsRemaining = fadeSteps

        onboardingMusicFadeTimer = Timer.scheduledTimer(withTimeInterval: stepInterval, repeats: true) { [weak self] timer in
            stepsRemaining -= 1
            player.volume -= volumeDecrement

            if stepsRemaining <= 0 {
                timer.invalidate()
                player.stop()
                self?.onboardingMusicPlayer = nil
                self?.onboardingMusicFadeTimer = nil
            }
        }
    }

    func clearDetectedElementLocation() {
        detectedElementScreenLocation = nil
        detectedElementDisplayFrame = nil
        detectedElementBubbleText = nil
        // Drop any click that never fired (animation cancelled, user spoke
        // again) so a stale click can never land on whatever is on screen later.
        pendingClickAction = nil
        pendingClickFallbackTask?.cancel()
        pendingClickFallbackTask = nil
    }

    // MARK: - Click Actions

    /// A real mouse click waiting for the overlay cursor to arrive at the
    /// target. Set when Claude returns a [CLICK:...] tag and the "Allow
    /// actions" toggle is on; performed by the overlay when the flight
    /// animation lands (see BlueCursorView.startPointingAtElement).
    private struct PendingClickAction {
        /// Where to click, in global AppKit screen coordinates (bottom-left origin).
        let globalScreenLocation: CGPoint
        /// Short element label from the tag, for logging.
        let elementLabel: String
    }

    private var pendingClickAction: PendingClickAction?
    /// Safety net: performs the pending click even if the overlay flight
    /// animation never lands (e.g. it was blocked by the welcome animation).
    private var pendingClickFallbackTask: Task<Void, Never>?

    /// Called by BlueCursorView the moment the buddy arrives at the pointed
    /// element, so the real click happens right when the animation lands.
    func performPendingClickAtPointedElementIfNeeded() {
        guard let clickAction = pendingClickAction else { return }
        pendingClickAction = nil
        pendingClickFallbackTask?.cancel()
        pendingClickFallbackTask = nil
        performSingleLeftClick(
            atGlobalAppKitPoint: clickAction.globalScreenLocation,
            elementLabel: clickAction.elementLabel
        )
    }

    /// Synthesizes one left mouse click at the given global AppKit point via
    /// CGEvent, then restores the user's real cursor to where it was. Only a
    /// single left click is ever synthesized — no double clicks, drags, or
    /// other buttons.
    private func performSingleLeftClick(atGlobalAppKitPoint globalAppKitPoint: CGPoint, elementLabel: String) {
        guard let primaryScreen = NSScreen.screens.first else { return }

        // AppKit global coordinates have a bottom-left origin on the primary
        // screen; CGEvent coordinates have a top-left origin on the primary
        // screen. Only the Y axis flips — X is shared between the two spaces.
        let clickPointInCGEventCoordinates = CGPoint(
            x: globalAppKitPoint.x,
            y: primaryScreen.frame.maxY - globalAppKitPoint.y
        )

        // Remember where the user's real cursor is so it can be put back —
        // the synthesized click warps the cursor to the click point.
        let originalCursorPositionInCGEventCoordinates = CGEvent(source: nil)?.location

        guard let mouseDownEvent = CGEvent(
                  mouseEventSource: nil,
                  mouseType: .leftMouseDown,
                  mouseCursorPosition: clickPointInCGEventCoordinates,
                  mouseButton: .left
              ),
              let mouseUpEvent = CGEvent(
                  mouseEventSource: nil,
                  mouseType: .leftMouseUp,
                  mouseCursorPosition: clickPointInCGEventCoordinates,
                  mouseButton: .left
              ) else {
            print("⚠️ Click action: failed to create CGEvents for \"\(elementLabel)\"")
            return
        }

        mouseDownEvent.post(tap: .cghidEventTap)
        mouseUpEvent.post(tap: .cghidEventTap)

        if let originalCursorPositionInCGEventCoordinates {
            CGWarpMouseCursorPosition(originalCursorPositionInCGEventCoordinates)
        }

        print("🖱️ Click action: clicked \"\(elementLabel)\" at (\(Int(globalAppKitPoint.x)), \(Int(globalAppKitPoint.y))) global AppKit coords")
    }

    func stop() {
        globalPushToTalkShortcutMonitor.stop()
        buddyDictationManager.cancelCurrentDictation()
        overlayWindowManager.hideOverlay()
        transientHideTask?.cancel()

        currentResponseTask?.cancel()
        currentResponseTask = nil
        shortcutTransitionCancellable?.cancel()
        voiceStateCancellable?.cancel()
        audioPowerCancellable?.cancel()
        accessibilityCheckTimer?.invalidate()
        accessibilityCheckTimer = nil
    }

    func refreshAllPermissions() {
        let previouslyHadAccessibility = hasAccessibilityPermission
        let previouslyHadScreenRecording = hasScreenRecordingPermission
        let previouslyHadMicrophone = hasMicrophonePermission
        let previouslyHadAll = allPermissionsGranted

        let currentlyHasAccessibility = WindowPositionManager.hasAccessibilityPermission()
        hasAccessibilityPermission = currentlyHasAccessibility

        if currentlyHasAccessibility {
            globalPushToTalkShortcutMonitor.start()
        } else {
            globalPushToTalkShortcutMonitor.stop()
        }

        hasScreenRecordingPermission = WindowPositionManager.hasScreenRecordingPermission()

        let micAuthStatus = AVCaptureDevice.authorizationStatus(for: .audio)
        hasMicrophonePermission = micAuthStatus == .authorized

        // Debug: log permission state on changes
        if previouslyHadAccessibility != hasAccessibilityPermission
            || previouslyHadScreenRecording != hasScreenRecordingPermission
            || previouslyHadMicrophone != hasMicrophonePermission {
            print("🔑 Permissions — accessibility: \(hasAccessibilityPermission), screen: \(hasScreenRecordingPermission), mic: \(hasMicrophonePermission), screenContent: \(hasScreenContentPermission)")
        }

        // Track individual permission grants as they happen
        if !previouslyHadAccessibility && hasAccessibilityPermission {
            ClickyAnalytics.trackPermissionGranted(permission: "accessibility")
        }
        if !previouslyHadScreenRecording && hasScreenRecordingPermission {
            ClickyAnalytics.trackPermissionGranted(permission: "screen_recording")
        }
        if !previouslyHadMicrophone && hasMicrophonePermission {
            ClickyAnalytics.trackPermissionGranted(permission: "microphone")
        }
        // Screen content permission is persisted — once the user has approved the
        // SCShareableContent picker, we don't need to re-check it.
        if !hasScreenContentPermission {
            hasScreenContentPermission = UserDefaults.standard.bool(forKey: "hasScreenContentPermission")
        }

        if !previouslyHadAll && allPermissionsGranted {
            ClickyAnalytics.trackAllPermissionsGranted()
        }
    }

    /// Triggers the macOS screen content picker by performing a dummy
    /// screenshot capture. Once the user approves, we persist the grant
    /// so they're never asked again during onboarding.
    @Published private(set) var isRequestingScreenContent = false

    func requestScreenContentPermission() {
        guard !isRequestingScreenContent else { return }
        isRequestingScreenContent = true
        Task {
            do {
                let content = try await SCShareableContent.excludingDesktopWindows(false, onScreenWindowsOnly: true)
                guard let display = content.displays.first else {
                    await MainActor.run { isRequestingScreenContent = false }
                    return
                }
                let filter = SCContentFilter(display: display, excludingWindows: [])
                let config = SCStreamConfiguration()
                config.width = 320
                config.height = 240
                let image = try await SCScreenshotManager.captureImage(contentFilter: filter, configuration: config)
                // Verify the capture actually returned real content — a 0x0 or
                // fully-empty image means the user denied the prompt.
                let didCapture = image.width > 0 && image.height > 0
                print("🔑 Screen content capture result — width: \(image.width), height: \(image.height), didCapture: \(didCapture)")
                await MainActor.run {
                    isRequestingScreenContent = false
                    guard didCapture else { return }
                    hasScreenContentPermission = true
                    UserDefaults.standard.set(true, forKey: "hasScreenContentPermission")
                    ClickyAnalytics.trackPermissionGranted(permission: "screen_content")

                    // If onboarding was already completed, show the cursor overlay now
                    if hasCompletedOnboarding && allPermissionsGranted && !isOverlayVisible && isClickyCursorEnabled {
                        overlayWindowManager.hasShownOverlayBefore = true
                        overlayWindowManager.showOverlay(onScreens: NSScreen.screens, companionManager: self)
                        isOverlayVisible = true
                    }
                }
            } catch {
                print("⚠️ Screen content permission request failed: \(error)")
                await MainActor.run { isRequestingScreenContent = false }
            }
        }
    }

    // MARK: - Private

    /// Triggers the system microphone prompt if the user has never been asked.
    /// Once granted/denied the status sticks and polling picks it up.
    private func promptForMicrophoneIfNotDetermined() {
        guard AVCaptureDevice.authorizationStatus(for: .audio) == .notDetermined else { return }
        AVCaptureDevice.requestAccess(for: .audio) { [weak self] granted in
            Task { @MainActor [weak self] in
                self?.hasMicrophonePermission = granted
            }
        }
    }

    /// Polls all permissions frequently so the UI updates live after the
    /// user grants them in System Settings. Screen Recording is the exception —
    /// macOS requires an app restart for that one to take effect.
    private func startPermissionPolling() {
        accessibilityCheckTimer = Timer.scheduledTimer(withTimeInterval: 1.5, repeats: true) { [weak self] _ in
            Task { @MainActor [weak self] in
                self?.refreshAllPermissions()
            }
        }
    }

    private func bindAudioPowerLevel() {
        audioPowerCancellable = buddyDictationManager.$currentAudioPowerLevel
            .receive(on: DispatchQueue.main)
            .sink { [weak self] powerLevel in
                self?.currentAudioPowerLevel = powerLevel
            }
    }

    private func bindVoiceStateObservation() {
        voiceStateCancellable = buddyDictationManager.$isRecordingFromKeyboardShortcut
            .combineLatest(
                buddyDictationManager.$isFinalizingTranscript,
                buddyDictationManager.$isPreparingToRecord
            )
            .receive(on: DispatchQueue.main)
            .sink { [weak self] isRecording, isFinalizing, isPreparing in
                guard let self else { return }
                // Don't override .responding — the AI response pipeline
                // manages that state directly until streaming finishes.
                guard self.voiceState != .responding else { return }

                if isFinalizing {
                    self.voiceState = .processing
                } else if isRecording {
                    self.voiceState = .listening
                } else if isPreparing {
                    self.voiceState = .processing
                } else {
                    self.voiceState = .idle
                    // If the user pressed and released the hotkey without
                    // saying anything, no response task runs — schedule the
                    // transient hide here so the overlay doesn't get stuck.
                    // Only do this when no response is in flight, otherwise
                    // the brief idle gap between recording and processing
                    // would prematurely hide the overlay.
                    if self.currentResponseTask == nil {
                        self.scheduleTransientHideIfNeeded()
                    }
                }
            }
    }

    private func bindShortcutTransitions() {
        shortcutTransitionCancellable = globalPushToTalkShortcutMonitor
            .shortcutTransitionPublisher
            .receive(on: DispatchQueue.main)
            .sink { [weak self] transition in
                self?.handleShortcutTransition(transition)
            }
    }

    private func handleShortcutTransition(_ transition: BuddyPushToTalkShortcut.ShortcutTransition) {
        switch transition {
        case .pressed:
            guard !buddyDictationManager.isDictationInProgress else { return }
            // Don't register push-to-talk while the onboarding video is playing
            guard !showOnboardingVideo else { return }

            // Cancel any pending transient hide so the overlay stays visible
            transientHideTask?.cancel()
            transientHideTask = nil

            // If the cursor is hidden, bring it back transiently for this interaction
            if !isClickyCursorEnabled && !isOverlayVisible {
                overlayWindowManager.hasShownOverlayBefore = true
                overlayWindowManager.showOverlay(onScreens: NSScreen.screens, companionManager: self)
                isOverlayVisible = true
            }

            // Dismiss the menu bar panel so it doesn't cover the screen
            NotificationCenter.default.post(name: .clickyDismissPanel, object: nil)

            // Cancel any in-progress response and TTS from a previous utterance
            currentResponseTask?.cancel()
            appleTTSClient.stopPlayback()
            kokoroTTSClient.stopPlayback()
            clearDetectedElementLocation()

            // Dismiss the onboarding prompt if it's showing
            if showOnboardingPrompt {
                withAnimation(.easeOut(duration: 0.3)) {
                    onboardingPromptOpacity = 0.0
                }
                DispatchQueue.main.asyncAfter(deadline: .now() + 0.35) {
                    self.showOnboardingPrompt = false
                    self.onboardingPromptText = ""
                }
            }
    

            ClickyAnalytics.trackPushToTalkStarted()

            pendingKeyboardShortcutStartTask?.cancel()
            pendingKeyboardShortcutStartTask = Task {
                await buddyDictationManager.startPushToTalkFromKeyboardShortcut(
                    currentDraftText: "",
                    updateDraftText: { _ in
                        // Partial transcripts are hidden (waveform-only UI)
                    },
                    submitDraftText: { [weak self] finalTranscript in
                        self?.lastTranscript = finalTranscript
                        print("🗣️ Companion received transcript: \(finalTranscript)")
                        ClickyAnalytics.trackUserMessageSent(transcript: finalTranscript)
                        self?.sendTranscriptToClaudeWithScreenshot(transcript: finalTranscript)
                    }
                )
            }
        case .released:
            // Cancel the pending start task in case the user released the shortcut
            // before the async startPushToTalk had a chance to begin recording.
            // Without this, a quick press-and-release drops the release event and
            // leaves the waveform overlay stuck on screen indefinitely.
            ClickyAnalytics.trackPushToTalkReleased()
            pendingKeyboardShortcutStartTask?.cancel()
            pendingKeyboardShortcutStartTask = nil
            buddyDictationManager.stopPushToTalkFromKeyboardShortcut()
        case .none:
            break
        }
    }

    // MARK: - Companion Prompt

    private static let companionVoiceResponseSystemPrompt = """
    you're clicky, a friendly always-on companion that lives in the user's menu bar. the user just spoke to you via push-to-talk and you can see their screen(s). your reply will be spoken aloud via text-to-speech, so write the way you'd actually talk. this is an ongoing conversation — you remember everything they've said before.

    rules:
    - default to one or two sentences. be direct and dense. BUT if the user asks you to explain more, go deeper, or elaborate, then go all out — give a thorough, detailed explanation with no length limit.
    - all lowercase, casual, warm. no emojis.
    - write for the ear, not the eye. short sentences. no lists, bullet points, markdown, or formatting — just natural speech.
    - don't use abbreviations or symbols that sound weird read aloud. write "for example" not "e.g.", spell out small numbers.
    - if the user's question relates to what's on their screen, reference specific things you see.
    - if the screenshot doesn't seem relevant to their question, just answer the question directly.
    - you can help with anything — coding, writing, general knowledge, brainstorming.
    - never say "simply" or "just".
    - don't read out code verbatim. describe what the code does or what needs to change conversationally.
    - focus on giving a thorough, useful explanation. don't end with simple yes/no questions like "want me to explain more?" or "should i show you?" — those are dead ends that force the user to just say yes.
    - instead, when it fits naturally, end by planting a seed — mention something bigger or more ambitious they could try, a related concept that goes deeper, or a next-level technique that builds on what you just explained. make it something worth coming back for, not a question they'd just nod to. it's okay to not end with anything extra if the answer is complete on its own.
    - if you receive multiple screen images, the one labeled "primary focus" is where the cursor is — prioritize that one but reference others if relevant.

    element pointing:
    you have a small blue triangle cursor that can fly to and point at things on screen. use it whenever pointing would genuinely help the user — if they're asking how to do something, looking for a menu, trying to find a button, or need help navigating an app, point at the relevant element. err on the side of pointing rather than not pointing, because it makes your help way more useful and concrete.

    don't point at things when it would be pointless — like if the user asks a general knowledge question, or the conversation has nothing to do with what's on screen, or you'd just be pointing at something obvious they're already looking at. but if there's a specific UI element, menu, button, or area on screen that's relevant to what you're helping with, point at it.

    when you point, append a coordinate tag at the very end of your response, AFTER your spoken text. the screenshot images are labeled with their pixel dimensions. use those dimensions as the coordinate space. the origin (0,0) is the top-left corner of the image. x increases rightward, y increases downward.

    format: [POINT:x,y:label] where x,y are integer pixel coordinates in the screenshot's coordinate space, and label is a short 1-3 word description of the element (like "search bar" or "save button"). if the element is on the cursor's screen you can omit the screen number. if the element is on a DIFFERENT screen, append :screenN where N is the screen number from the image label (e.g. :screen2). this is important — without the screen number, the cursor will point at the wrong place.

    if pointing wouldn't help, append [POINT:none].

    click actions:
    you can also perform a REAL mouse click for the user. emit a CLICK tag ONLY when the user's request is explicitly an action command — they ask you to click, press, open, or select something FOR them ("click the save button", "open that menu", "select the first result", "press submit for me"). for informational questions ("where is the save button?", "how do i open settings?") use POINT as before, never CLICK. when in doubt, use POINT — a wrong click is worse than a wrong point.

    strict CLICK rules:
    - format: [CLICK:x,y:label] — or [CLICK:x,y:label:screenN] if the element is on a different screen than the cursor. same coordinate space rules as POINT.
    - never emit both a POINT tag and a CLICK tag in the same response — exactly one tag, and it must be the very last thing in your response.
    - when you emit CLICK, keep the spoken text to a brief confirmation of the action, like "clicking the save button."

    multi-step tasks:
    if the user asks you to DO something that needs more than one action to complete — like "open safari and search for flights", "create a new note and write my shopping list in it", "reply to that email saying i'll be there" — do not try to cram it into a single CLICK. instead, speak a very brief confirmation of the plan (like "on it, opening safari and searching for flights") and end with [TASK] as your tag. clicky will then work through the task one step at a time, clicking and typing as needed.
    strict TASK rules:
    - [TASK] is only for explicit action requests that need multiple steps. questions are never TASK. a single simple click is still CLICK, not TASK.
    - [TASK] takes no coordinates — it's exactly the literal text [TASK] as the very last thing in your response, and like the other tags it replaces POINT/CLICK (never emit two tags).

    examples:
    - user asks how to color grade in final cut: "you'll want to open the color inspector — it's right up in the top right area of the toolbar. click that and you'll get all the color wheels and curves. [POINT:1100,42:color inspector]"
    - user asks what html is: "html stands for hypertext markup language, it's basically the skeleton of every web page. curious how it connects to the css you're looking at? [POINT:none]"
    - user asks how to commit in xcode: "see that source control menu up top? click that and hit commit, or you can use command option c as a shortcut. [POINT:285,11:source control]"
    - element is on screen 2 (not where cursor is): "that's over on your other monitor — see the terminal window? [POINT:400,300:terminal:screen2]"
    - user says "click the save button": "clicking the save button. [CLICK:640,52:save button]"
    - user says "where's the save button?" (informational, so POINT not CLICK): "it's up in the top right of the toolbar. [POINT:640,52:save button]"
    - user says "open the file menu for me": "opening the file menu. [CLICK:85,11:file menu]"
    - user says "open safari and go to gmail": "on it, opening safari and heading to gmail. [TASK]"
    - user says "make a new note and title it groceries": "sure, making a new note called groceries. [TASK]"
    """

    // MARK: - AI Response Pipeline

    /// Captures a screenshot, sends it along with the transcript to Claude,
    /// and speaks the response aloud via the preferred TTS voice (Kokoro by
    /// default, Apple system voice as fallback). The cursor stays in
    /// the spinner/processing state until TTS audio begins playing.
    /// Claude's response may include a [POINT:x,y:label] tag which triggers
    /// the buddy to fly to that element on screen.
    private func sendTranscriptToClaudeWithScreenshot(transcript: String) {
        currentResponseTask?.cancel()
        appleTTSClient.stopPlayback()
        kokoroTTSClient.stopPlayback()

        currentResponseTask = Task {
            // Stay in processing (spinner) state — no streaming text displayed
            voiceState = .processing

            do {
                // Capture all connected screens so the AI has full context
                let screenCaptures = try await CompanionScreenCaptureUtility.captureAllScreensAsJPEG()

                guard !Task.isCancelled else { return }

                // Build image labels with the actual screenshot pixel dimensions
                // so Claude's coordinate space matches the image it sees. We
                // scale from screenshot pixels to display points ourselves.
                let labeledImages = screenCaptures.map { capture in
                    let dimensionInfo = " (image dimensions: \(capture.screenshotWidthInPixels)x\(capture.screenshotHeightInPixels) pixels)"
                    return (data: capture.imageData, label: capture.label + dimensionInfo)
                }

                // Pass conversation history so Claude remembers prior exchanges
                let historyForAPI = conversationHistory.map { entry in
                    (userPlaceholder: entry.userTranscript, assistantResponse: entry.assistantResponse)
                }

                let (fullResponseText, _) = try await claudeAPI.analyzeImageStreaming(
                    images: labeledImages,
                    systemPrompt: Self.companionVoiceResponseSystemPrompt,
                    conversationHistory: historyForAPI,
                    userPrompt: transcript,
                    onTextChunk: { _ in
                        // No streaming text display — spinner stays until TTS plays
                    }
                )

                guard !Task.isCancelled else { return }

                // Multi-step task hand-off: a trailing [TASK] tag means this
                // request needs the step-by-step agent loop (multiple clicks,
                // typing, key presses) rather than a single POINT/CLICK.
                let agentTaskParseResult = Self.parseAgentTaskTag(from: fullResponseText)
                if agentTaskParseResult.isAgentTask {
                    await runAgentTask(
                        taskGoal: transcript,
                        acknowledgmentSpokenText: agentTaskParseResult.spokenText
                    )
                    if !Task.isCancelled {
                        voiceState = .idle
                        scheduleTransientHideIfNeeded()
                    }
                    return
                }

                // Parse the [POINT:...] / [CLICK:...] tag from Claude's response
                let parseResult = Self.parsePointingCoordinates(from: fullResponseText)
                let spokenText = parseResult.spokenText

                // A CLICK tag only performs a real click when the "Allow
                // actions" toggle is on — otherwise it degrades to pointing.
                let shouldPerformClickAction = parseResult.isClickAction && allowClickActions

                // Pick the screen capture matching Claude's screen number,
                // falling back to the cursor screen if not specified.
                let targetScreenCapture: CompanionScreenCapture? = {
                    if let screenNumber = parseResult.screenNumber,
                       screenNumber >= 1 && screenNumber <= screenCaptures.count {
                        return screenCaptures[screenNumber - 1]
                    }
                    return screenCaptures.first(where: { $0.isCursorScreen })
                }()

                if var pointCoordinate = parseResult.coordinate,
                   let targetScreenCapture {
                    // Precision second pass: re-ask Claude (via the locator)
                    // for exact coordinates using the SAME screenshot. Always
                    // refine before clicking — a real click must be accurate.
                    // For pointing, refine only when "PrecisePointing" is on.
                    // Only meaningful in subscription mode — the locator talks
                    // to the local bridge, which API-key users don't run.
                    let shouldRefineCoordinate = (shouldPerformClickAction || isPrecisePointingEnabled)
                        && claudeBackendMode == .subscription
                    if shouldRefineCoordinate, let elementLabel = parseResult.elementLabel {
                        if let refinedCoordinate = await elementLocationDetector.refineElementCoordinate(
                            screenshotData: targetScreenCapture.imageData,
                            screenshotWidthInPixels: targetScreenCapture.screenshotWidthInPixels,
                            screenshotHeightInPixels: targetScreenCapture.screenshotHeightInPixels,
                            elementLabel: elementLabel,
                            initialCoordinate: pointCoordinate
                        ) {
                            pointCoordinate = refinedCoordinate
                        }
                        guard !Task.isCancelled else { return }
                    }

                    // Switch to idle BEFORE setting the location so the triangle
                    // becomes visible and can fly to the target. Without this, the
                    // spinner hides the triangle and the flight animation is invisible.
                    voiceState = .idle

                    // Claude's coordinates are in the screenshot's pixel space
                    // (top-left origin, e.g. 1568x980). Scale to the display's
                    // point space (e.g. 1512x982), then convert to AppKit global coords.
                    let displayFrame = targetScreenCapture.displayFrame
                    let globalLocation = Self.mapScreenshotPixelCoordinateToGlobalAppKitPoint(
                        pointCoordinate,
                        on: targetScreenCapture
                    )

                    if shouldPerformClickAction {
                        // Safety: only click a point that actually lies on a
                        // connected screen — never synthesize an off-screen click.
                        let clickTargetIsOnAScreen = NSScreen.screens.contains { screen in
                            screen.frame.contains(globalLocation)
                        }
                        if clickTargetIsOnAScreen {
                            // The overlay performs the click when the flight
                            // animation lands on the target (so the user sees the
                            // buddy arrive first). The fallback fires it anyway if
                            // the animation never lands — flights max out at 1.4s.
                            pendingClickAction = PendingClickAction(
                                globalScreenLocation: globalLocation,
                                elementLabel: parseResult.elementLabel ?? "element"
                            )
                            pendingClickFallbackTask?.cancel()
                            pendingClickFallbackTask = Task { [weak self] in
                                try? await Task.sleep(nanoseconds: 3_000_000_000)
                                guard !Task.isCancelled else { return }
                                self?.performPendingClickAtPointedElementIfNeeded()
                            }
                        } else {
                            print("⚠️ Click action: target (\(Int(globalLocation.x)), \(Int(globalLocation.y))) is outside every screen — pointing only")
                        }
                    }

                    detectedElementScreenLocation = globalLocation
                    detectedElementDisplayFrame = displayFrame
                    ClickyAnalytics.trackElementPointed(elementLabel: parseResult.elementLabel)
                    print("🎯 Element \(shouldPerformClickAction ? "click" : "pointing"): (\(Int(pointCoordinate.x)), \(Int(pointCoordinate.y))) → \"\(parseResult.elementLabel ?? "element")\"")
                } else {
                    print("🎯 Element pointing: \(parseResult.elementLabel ?? "no element")")
                }

                // Save this exchange to conversation history (with the point tag
                // stripped so it doesn't confuse future context)
                conversationHistory.append((
                    userTranscript: transcript,
                    assistantResponse: spokenText
                ))

                // Keep only the last 10 exchanges to avoid unbounded context growth
                if conversationHistory.count > 10 {
                    conversationHistory.removeFirst(conversationHistory.count - 10)
                }

                print("🧠 Conversation history: \(conversationHistory.count) exchanges")

                ClickyAnalytics.trackAIResponseReceived(response: spokenText)

                // Play the response via TTS. Keep the spinner (processing state)
                // until the audio actually starts playing, then switch to responding.
                if !spokenText.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
                    do {
                        try await speakResponseText(spokenText)
                        // speakResponseText returns once audio is playing
                        voiceState = .responding
                    } catch {
                        ClickyAnalytics.trackTTSError(error: error.localizedDescription)
                        print("⚠️ TTS error: \(error)")
                        speakResponseErrorFallback(for: error)
                    }
                }
            } catch is CancellationError {
                // User spoke again — response was interrupted
            } catch {
                ClickyAnalytics.trackResponseError(error: error.localizedDescription)
                print("⚠️ Companion response error: \(error)")
                speakResponseErrorFallback(for: error)
            }

            if !Task.isCancelled {
                voiceState = .idle
                scheduleTransientHideIfNeeded()
            }
        }
    }

    /// Speaks a response using the preferred voice. Tries Kokoro first when
    /// the "PreferKokoroVoice" setting is on, and falls back to the Apple
    /// system voice if Kokoro fails for any reason (models still downloading
    /// on first run, synthesis error) so the user always hears a response.
    private func speakResponseText(_ spokenText: String) async throws {
        if preferKokoroVoice {
            do {
                try await kokoroTTSClient.speakText(spokenText)
                return
            } catch is CancellationError {
                throw CancellationError()
            } catch {
                print("⚠️ Kokoro TTS unavailable (\(error.localizedDescription)) — falling back to Apple TTS for this utterance")
            }
        }
        try await appleTTSClient.speakText(spokenText)
    }

    /// If the cursor is in transient mode (user toggled "Show Clicky" off),
    /// waits for TTS playback and any pointing animation to finish, then
    /// fades out the overlay after a 1-second pause. Cancelled automatically
    /// if the user starts another push-to-talk interaction.
    private func scheduleTransientHideIfNeeded() {
        guard !isClickyCursorEnabled && isOverlayVisible else { return }

        transientHideTask?.cancel()
        transientHideTask = Task {
            // Wait for TTS audio to finish playing (whichever voice spoke)
            while appleTTSClient.isPlaying || kokoroTTSClient.isPlaying {
                try? await Task.sleep(nanoseconds: 200_000_000)
                guard !Task.isCancelled else { return }
            }

            // Wait for pointing animation to finish (location is cleared
            // when the buddy flies back to the cursor)
            while detectedElementScreenLocation != nil {
                try? await Task.sleep(nanoseconds: 200_000_000)
                guard !Task.isCancelled else { return }
            }

            // Pause 1s after everything finishes, then fade out
            try? await Task.sleep(nanoseconds: 1_000_000_000)
            guard !Task.isCancelled else { return }
            overlayWindowManager.fadeOutAndHideOverlay()
            isOverlayVisible = false
        }
    }

    /// Speaks a short error message using macOS system TTS when the response
    /// pipeline fails. Uses NSSpeechSynthesizer so it works even when the
    /// primary TTS path is unavailable. The message is chosen from the error:
    /// timeout, bridge unreachable, missing API key, or a generic model error —
    /// kept short because it's spoken aloud.
    private func speakResponseErrorFallback(for responseError: Error) {
        let utterance: String
        let errorCode = (responseError as NSError).code
        let errorDescription = (responseError as NSError).localizedDescription

        if errorCode == URLError.timedOut.rawValue {
            utterance = "That took too long, try again."
        } else if claudeBackendMode == .subscription
                    && (errorDescription.contains("local bridge") || errorDescription.contains("127.0.0.1:8377")) {
            // ClaudeAPI maps connection failures to a "local bridge" guidance
            // error in subscription mode (see mapTransportErrorToBridgeGuidanceIfNeeded).
            utterance = "The local bridge isn't running."
        } else if claudeBackendMode == .apiKey && !hasAnthropicAPIKey {
            utterance = "I need an Anthropic API key. Open the Clicky panel in the menu bar and paste one in."
        } else {
            utterance = "Claude couldn't process that."
        }
        let synthesizer = NSSpeechSynthesizer()
        synthesizer.startSpeaking(utterance)
        voiceState = .responding
    }

    // MARK: - Point/Click Tag Parsing

    /// Result of parsing a [POINT:...] or [CLICK:...] tag from Claude's response.
    struct PointingParseResult {
        /// The response text with the tag removed — this is what gets spoken.
        let spokenText: String
        /// The parsed pixel coordinate, or nil if Claude said "none" or no tag was found.
        let coordinate: CGPoint?
        /// Short label describing the element (e.g. "run button"), or "none".
        let elementLabel: String?
        /// Which screen the coordinate refers to (1-based), or nil to default to cursor screen.
        let screenNumber: Int?
        /// True when the tag was [CLICK:...] — the user asked for a real click,
        /// not just pointing. Whether the click actually happens also depends
        /// on the "Allow actions" toggle.
        let isClickAction: Bool
    }

    /// Parses a [POINT:x,y:label:screenN], [CLICK:x,y:label:screenN], or
    /// [POINT:none] tag from the end of Claude's response. Returns the spoken
    /// text (tag removed) plus the optional coordinate + label + screen number,
    /// and whether the tag requests a real click.
    static func parsePointingCoordinates(from responseText: String) -> PointingParseResult {
        // Match [POINT:none] / [POINT:123,456:label(:screen2)] / [CLICK:123,456:label(:screen2)]
        let pattern = #"\[(POINT|CLICK):(?:none|(\d+)\s*,\s*(\d+)(?::([^\]:\s][^\]:]*?))?(?::screen(\d+))?)\]\s*$"#

        guard let regex = try? NSRegularExpression(pattern: pattern, options: []),
              let match = regex.firstMatch(in: responseText, range: NSRange(responseText.startIndex..., in: responseText)) else {
            // No tag found at all
            return PointingParseResult(spokenText: responseText, coordinate: nil, elementLabel: nil, screenNumber: nil, isClickAction: false)
        }

        // Remove the tag from the spoken text
        let tagRange = Range(match.range, in: responseText)!
        let spokenText = String(responseText[..<tagRange.lowerBound]).trimmingCharacters(in: .whitespacesAndNewlines)

        var isClickAction = false
        if let tagKindRange = Range(match.range(at: 1), in: responseText) {
            isClickAction = responseText[tagKindRange] == "CLICK"
        }

        // Check if it's [POINT:none] (or a malformed coordinate)
        guard match.numberOfRanges >= 4,
              let xRange = Range(match.range(at: 2), in: responseText),
              let yRange = Range(match.range(at: 3), in: responseText),
              let x = Double(responseText[xRange]),
              let y = Double(responseText[yRange]) else {
            return PointingParseResult(spokenText: spokenText, coordinate: nil, elementLabel: "none", screenNumber: nil, isClickAction: false)
        }

        var elementLabel: String? = nil
        if match.numberOfRanges >= 5, let labelRange = Range(match.range(at: 4), in: responseText) {
            elementLabel = String(responseText[labelRange]).trimmingCharacters(in: .whitespaces)
        }

        var screenNumber: Int? = nil
        if match.numberOfRanges >= 6, let screenRange = Range(match.range(at: 5), in: responseText) {
            screenNumber = Int(responseText[screenRange])
        }

        return PointingParseResult(
            spokenText: spokenText,
            coordinate: CGPoint(x: x, y: y),
            elementLabel: elementLabel,
            screenNumber: screenNumber,
            isClickAction: isClickAction
        )
    }

    // MARK: - Multi-Step Agent Tasks

    struct AgentTaskParseResult {
        /// True when the response ended with a [TASK] tag — the request needs
        /// the multi-step agent loop instead of a single POINT/CLICK.
        let isAgentTask: Bool
        /// The response text with the [TASK] tag removed — spoken as the
        /// acknowledgment before the agent loop starts.
        let spokenText: String
    }

    /// Detects a trailing [TASK] tag in Claude's response. Runs before
    /// parsePointingCoordinates because a TASK response has no coordinates.
    static func parseAgentTaskTag(from responseText: String) -> AgentTaskParseResult {
        let pattern = #"\[TASK\]\s*$"#
        guard let regex = try? NSRegularExpression(pattern: pattern),
              let match = regex.firstMatch(in: responseText, range: NSRange(responseText.startIndex..., in: responseText)),
              let tagRange = Range(match.range, in: responseText) else {
            return AgentTaskParseResult(isAgentTask: false, spokenText: responseText)
        }
        let spokenText = String(responseText[..<tagRange.lowerBound]).trimmingCharacters(in: .whitespacesAndNewlines)
        return AgentTaskParseResult(isAgentTask: true, spokenText: spokenText)
    }

    /// One action the agent loop can perform per step, parsed from the
    /// action tag at the end of an agent-step response.
    enum AgentStepAction {
        case click(coordinate: CGPoint, elementLabel: String, screenNumber: Int?)
        case typeText(String)
        case pressKey(String)
        case scroll(direction: AgentActionExecutor.ScrollDirection)
        case done(summary: String)
        case fail(reason: String)
    }

    struct AgentStepParseResult {
        /// The response text with the action tag removed — spoken as the
        /// step's narration ("opening spotlight").
        let narrationText: String
        /// The parsed action, or nil when the response had no valid tag.
        let action: AgentStepAction?
    }

    /// Parses the single action tag at the end of an agent-step response:
    /// [CLICK:x,y:label(:screenN)], [TYPE:text], [KEY:combo], [SCROLL:up|down],
    /// [DONE:summary], or [FAIL:reason].
    static func parseAgentStepAction(from responseText: String) -> AgentStepParseResult {
        let pattern = #"\[(CLICK|TYPE|KEY|SCROLL|DONE|FAIL)(?::([\s\S]*?))?\]\s*$"#
        guard let regex = try? NSRegularExpression(pattern: pattern),
              let match = regex.firstMatch(in: responseText, range: NSRange(responseText.startIndex..., in: responseText)),
              let tagRange = Range(match.range, in: responseText),
              let tagNameRange = Range(match.range(at: 1), in: responseText) else {
            return AgentStepParseResult(
                narrationText: responseText.trimmingCharacters(in: .whitespacesAndNewlines),
                action: nil
            )
        }

        let narrationText = String(responseText[..<tagRange.lowerBound]).trimmingCharacters(in: .whitespacesAndNewlines)
        let tagName = String(responseText[tagNameRange])
        var tagPayload = ""
        if let payloadRange = Range(match.range(at: 2), in: responseText) {
            tagPayload = String(responseText[payloadRange])
        }

        switch tagName {
        case "CLICK":
            // Payload shape: "x,y:label" or "x,y:label:screenN"
            let payloadParts = tagPayload.split(separator: ":", omittingEmptySubsequences: false).map(String.init)
            guard let coordinatePart = payloadParts.first else {
                return AgentStepParseResult(narrationText: narrationText, action: nil)
            }
            let coordinatePieces = coordinatePart.split(separator: ",")
            guard coordinatePieces.count == 2,
                  let x = Double(coordinatePieces[0].trimmingCharacters(in: .whitespaces)),
                  let y = Double(coordinatePieces[1].trimmingCharacters(in: .whitespaces)) else {
                return AgentStepParseResult(narrationText: narrationText, action: nil)
            }
            var elementLabel = "element"
            if payloadParts.count >= 2 {
                let labelCandidate = payloadParts[1].trimmingCharacters(in: .whitespaces)
                if !labelCandidate.isEmpty { elementLabel = labelCandidate }
            }
            var screenNumber: Int? = nil
            if payloadParts.count >= 3, payloadParts[2].hasPrefix("screen") {
                screenNumber = Int(payloadParts[2].dropFirst("screen".count))
            }
            return AgentStepParseResult(
                narrationText: narrationText,
                action: .click(coordinate: CGPoint(x: x, y: y), elementLabel: elementLabel, screenNumber: screenNumber)
            )
        case "TYPE":
            return AgentStepParseResult(narrationText: narrationText, action: .typeText(tagPayload))
        case "KEY":
            return AgentStepParseResult(
                narrationText: narrationText,
                action: .pressKey(tagPayload.trimmingCharacters(in: .whitespaces))
            )
        case "SCROLL":
            let scrollDirection: AgentActionExecutor.ScrollDirection =
                tagPayload.lowercased().hasPrefix("up") ? .up : .down
            return AgentStepParseResult(narrationText: narrationText, action: .scroll(direction: scrollDirection))
        case "DONE":
            return AgentStepParseResult(
                narrationText: narrationText,
                action: .done(summary: tagPayload.trimmingCharacters(in: .whitespacesAndNewlines))
            )
        case "FAIL":
            return AgentStepParseResult(
                narrationText: narrationText,
                action: .fail(reason: tagPayload.trimmingCharacters(in: .whitespacesAndNewlines))
            )
        default:
            return AgentStepParseResult(narrationText: narrationText, action: nil)
        }
    }

    /// System prompt for each step of the agent loop. The <agent_step> marker
    /// tells the local bridge to skip the voice-style instruction and inject
    /// the agent-step reminder instead of the pointing reminder.
    private static let agentStepSystemPrompt = """
    <agent_step>
    you're clicky, operating the user's mac by hand to complete a task they asked for. each turn you see the current state of their screen(s), the task goal, and the steps already performed. decide the SINGLE next action that moves the task forward.

    respond with a tiny spoken narration (two to six lowercase words, like "opening spotlight" or "typing the address") followed by exactly ONE action tag as the very last thing in your response.

    action tags:
    - [CLICK:x,y:label] — one left click on an element. x,y are integer pixel coordinates in the labeled screenshot's coordinate space, origin top-left, x rightward, y downward. label is a short 1-3 word element name. if the element is on a different screen than the cursor, use [CLICK:x,y:label:screenN] with N from the image label.
    - [TYPE:the text to type] — types into whatever field currently has keyboard focus. click the field in an earlier step first if it isn't focused. never put square brackets inside the text.
    - [KEY:combo] — presses a key or shortcut. plain keys: return, escape, tab, space, delete, up, down, left, right, pageup, pagedown. combos with cmd, shift, option, ctrl — like cmd+t or cmd+shift+n.
    - [SCROLL:up] or [SCROLL:down] — scrolls the content in the middle of the screen.
    - [DONE:short spoken wrap-up] — the goal is achieved (or already was). the wrap-up is spoken aloud, one friendly lowercase sentence.
    - [FAIL:short spoken reason] — you cannot proceed (needed app missing, unexpected screen, login required). spoken aloud.

    rules:
    - exactly one action per turn. never chain two actions in one response.
    - prefer reliable keyboard routes: cmd+space, typing an app name, then return is the best way to open any app. return submits searches and forms. cmd+t opens a browser tab, cmd+l focuses the address bar.
    - before typing, make sure the target field has keyboard focus — click it first if unsure.
    - look at the screenshots carefully. if your previous step didn't have the expected effect, recover — do the right thing now instead of repeating the same action.
    - if the screen already shows the goal achieved, emit DONE instead of acting again.
    - never click anything destructive — delete, send, buy, submit payment — unless the task goal explicitly asks for exactly that.
    - the action tag must be the very last thing in your response, with nothing after it.
    """

    /// Converts a coordinate in a screen capture's screenshot pixel space
    /// (top-left origin) into global AppKit screen coordinates (bottom-left
    /// origin), scaling from screenshot pixels to display points. Used by both
    /// the one-shot POINT/CLICK path and the multi-step agent loop.
    private static func mapScreenshotPixelCoordinateToGlobalAppKitPoint(
        _ screenshotPixelCoordinate: CGPoint,
        on screenCapture: CompanionScreenCapture
    ) -> CGPoint {
        let screenshotWidth = CGFloat(screenCapture.screenshotWidthInPixels)
        let screenshotHeight = CGFloat(screenCapture.screenshotHeightInPixels)
        let displayWidth = CGFloat(screenCapture.displayWidthInPoints)
        let displayHeight = CGFloat(screenCapture.displayHeightInPoints)
        let displayFrame = screenCapture.displayFrame

        // Clamp to screenshot coordinate space
        let clampedX = max(0, min(screenshotPixelCoordinate.x, screenshotWidth))
        let clampedY = max(0, min(screenshotPixelCoordinate.y, screenshotHeight))

        // Scale from screenshot pixels to display points
        let displayLocalX = clampedX * (displayWidth / screenshotWidth)
        let displayLocalY = clampedY * (displayHeight / screenshotHeight)

        // Convert from top-left origin (screenshot) to bottom-left origin (AppKit)
        let appKitY = displayHeight - displayLocalY

        return CGPoint(
            x: displayLocalX + displayFrame.origin.x,
            y: appKitY + displayFrame.origin.y
        )
    }

    /// Works through a multi-step task one action at a time: capture the
    /// screen, ask Claude for the single next action, narrate it aloud,
    /// perform it, wait for the UI to settle, repeat — until [DONE], [FAIL],
    /// or the step cap. Runs inside currentResponseTask, so pressing the
    /// push-to-talk shortcut cancels the loop instantly (the press handler
    /// cancels currentResponseTask).
    private func runAgentTask(taskGoal: String, acknowledgmentSpokenText: String) async {
        // The agent loop clicks and types on the user's machine, so it is
        // gated behind the same "Allow actions" toggle as single clicks.
        guard allowClickActions else {
            let actionsDisabledMessage = "i'd love to, but actions are switched off. flip on allow actions in my menu bar panel and ask me again."
            try? await speakResponseText(actionsDisabledMessage)
            conversationHistory.append((userTranscript: taskGoal, assistantResponse: actionsDisabledMessage))
            return
        }

        if !acknowledgmentSpokenText.isEmpty {
            // speakResponseText returns once audio starts playing, so the
            // first step's screenshot happens while the acknowledgment plays.
            try? await speakResponseText(acknowledgmentSpokenText)
        }

        let maximumAgentStepCount = 10
        var completedStepDescriptions: [String] = []
        var finalOutcomeSpokenText: String? = nil

        agentLoop: for _ in 1...maximumAgentStepCount {
            guard !Task.isCancelled else { return }
            voiceState = .processing

            let stepResponseText: String
            let screenCaptures: [CompanionScreenCapture]
            do {
                screenCaptures = try await CompanionScreenCaptureUtility.captureAllScreensAsJPEG()
                guard !Task.isCancelled else { return }

                let labeledImages = screenCaptures.map { capture in
                    let dimensionInfo = " (image dimensions: \(capture.screenshotWidthInPixels)x\(capture.screenshotHeightInPixels) pixels)"
                    return (data: capture.imageData, label: capture.label + dimensionInfo)
                }

                let stepsPerformedText = completedStepDescriptions.isEmpty
                    ? "(none yet — this is the first step)"
                    : completedStepDescriptions.enumerated()
                        .map { "\($0.offset + 1). \($0.element)" }
                        .joined(separator: "\n")

                let stepUserPrompt = """
                <task_goal>\(taskGoal)</task_goal>
                <steps_already_performed>
                \(stepsPerformedText)
                </steps_already_performed>
                the screenshots show the CURRENT state of the screen(s), after the steps above. decide the single next action. if the goal is already achieved, emit [DONE:...].
                """

                (stepResponseText, _) = try await claudeAPI.analyzeImage(
                    images: labeledImages,
                    systemPrompt: Self.agentStepSystemPrompt,
                    userPrompt: stepUserPrompt
                )
            } catch is CancellationError {
                return
            } catch {
                print("⚠️ Agent task step error: \(error)")
                speakResponseErrorFallback(for: error)
                return
            }
            guard !Task.isCancelled else { return }

            let stepParseResult = Self.parseAgentStepAction(from: stepResponseText)
            guard let stepAction = stepParseResult.action else {
                // The model broke protocol (no action tag). Speak whatever it
                // said and stop rather than guessing at an action.
                finalOutcomeSpokenText = stepParseResult.narrationText.isEmpty
                    ? "i lost track of the next step, so i stopped. check the screen and ask me to continue."
                    : stepParseResult.narrationText
                break agentLoop
            }

            switch stepAction {
            case .done(let taskSummary):
                finalOutcomeSpokenText = taskSummary.isEmpty ? "all done." : taskSummary
                break agentLoop

            case .fail(let failureReason):
                finalOutcomeSpokenText = failureReason.isEmpty ? "i couldn't finish that task." : failureReason
                break agentLoop

            case .click(var clickCoordinate, let elementLabel, let screenNumber):
                if !stepParseResult.narrationText.isEmpty {
                    try? await speakResponseText(stepParseResult.narrationText)
                }

                let targetScreenCapture: CompanionScreenCapture? = {
                    if let screenNumber, screenNumber >= 1 && screenNumber <= screenCaptures.count {
                        return screenCaptures[screenNumber - 1]
                    }
                    return screenCaptures.first(where: { $0.isCursorScreen })
                }()
                guard let targetScreenCapture else {
                    finalOutcomeSpokenText = "i couldn't work out which screen to click on, so i stopped."
                    break agentLoop
                }

                // Precision second pass — a real click must be accurate.
                // Subscription mode only: the locator talks to the local bridge.
                if claudeBackendMode == .subscription {
                    if let refinedCoordinate = await elementLocationDetector.refineElementCoordinate(
                        screenshotData: targetScreenCapture.imageData,
                        screenshotWidthInPixels: targetScreenCapture.screenshotWidthInPixels,
                        screenshotHeightInPixels: targetScreenCapture.screenshotHeightInPixels,
                        elementLabel: elementLabel,
                        initialCoordinate: clickCoordinate
                    ) {
                        clickCoordinate = refinedCoordinate
                    }
                    guard !Task.isCancelled else { return }
                }

                let globalClickLocation = Self.mapScreenshotPixelCoordinateToGlobalAppKitPoint(
                    clickCoordinate,
                    on: targetScreenCapture
                )
                // Safety: never synthesize an off-screen click.
                let clickTargetIsOnAScreen = NSScreen.screens.contains { screen in
                    screen.frame.contains(globalClickLocation)
                }
                guard clickTargetIsOnAScreen else {
                    finalOutcomeSpokenText = "the next click would have landed off screen, so i stopped to be safe."
                    break agentLoop
                }

                // Fly the overlay cursor to the target so the user sees where
                // the click is about to land, then perform the real click.
                voiceState = .idle
                detectedElementScreenLocation = globalClickLocation
                detectedElementDisplayFrame = targetScreenCapture.displayFrame
                do { try await Task.sleep(nanoseconds: 1_200_000_000) } catch { return }
                performSingleLeftClick(atGlobalAppKitPoint: globalClickLocation, elementLabel: elementLabel)
                completedStepDescriptions.append("clicked \(elementLabel)")

            case .typeText(let textToType):
                if !stepParseResult.narrationText.isEmpty {
                    try? await speakResponseText(stepParseResult.narrationText)
                }
                AgentActionExecutor.typeText(textToType)
                completedStepDescriptions.append("typed \"\(textToType)\"")

            case .pressKey(let keyComboDescription):
                if !stepParseResult.narrationText.isEmpty {
                    try? await speakResponseText(stepParseResult.narrationText)
                }
                if AgentActionExecutor.pressKeyCombo(keyComboDescription) {
                    completedStepDescriptions.append("pressed \(keyComboDescription)")
                } else {
                    completedStepDescriptions.append("tried to press \(keyComboDescription) but that key isn't supported")
                }

            case .scroll(let scrollDirection):
                if !stepParseResult.narrationText.isEmpty {
                    try? await speakResponseText(stepParseResult.narrationText)
                }
                // Scroll events land on the window under the pointer, so aim
                // at the center of the cursor's screen.
                let scrollTargetScreenFrame = (screenCaptures.first(where: { $0.isCursorScreen }) ?? screenCaptures.first)?.displayFrame
                    ?? NSScreen.screens.first?.frame
                    ?? .zero
                if let primaryScreen = NSScreen.screens.first {
                    let scrollCenterInCGEventCoordinates = CGPoint(
                        x: scrollTargetScreenFrame.midX,
                        y: primaryScreen.frame.maxY - scrollTargetScreenFrame.midY
                    )
                    AgentActionExecutor.scroll(
                        atGlobalCGEventPoint: scrollCenterInCGEventCoordinates,
                        direction: scrollDirection
                    )
                }
                completedStepDescriptions.append("scrolled \(scrollDirection == .up ? "up" : "down")")
            }

            // Let the UI settle before the next screenshot so the model sees
            // the effect of the action it just took.
            do { try await Task.sleep(nanoseconds: 1_000_000_000) } catch { return }
        }

        let outcomeSpokenText = finalOutcomeSpokenText
            ?? "i've done \(maximumAgentStepCount) steps and the task isn't finished yet. take a look at where things are and tell me how to continue."

        // Save the whole task as one exchange so follow-up questions have context.
        conversationHistory.append((
            userTranscript: taskGoal,
            assistantResponse: acknowledgmentSpokenText.isEmpty
                ? outcomeSpokenText
                : acknowledgmentSpokenText + " " + outcomeSpokenText
        ))
        if conversationHistory.count > 10 {
            conversationHistory.removeFirst(conversationHistory.count - 10)
        }

        if !Task.isCancelled {
            try? await speakResponseText(outcomeSpokenText)
            voiceState = .responding
        }
    }

    // MARK: - Onboarding Video

    /// Sets up the onboarding video player, starts playback, and schedules
    /// the demo interaction at 40s. Called by BlueCursorView when onboarding starts.
    func setupOnboardingVideo() {
        guard let videoURL = URL(string: "https://stream.mux.com/e5jB8UuSrtFABVnTHCR7k3sIsmcUHCyhtLu1tzqLlfs.m3u8") else { return }

        let player = AVPlayer(url: videoURL)
        player.isMuted = false
        player.volume = 0.0
        self.onboardingVideoPlayer = player
        self.showOnboardingVideo = true
        self.onboardingVideoOpacity = 0.0

        // Start playback immediately — the video plays while invisible,
        // then we fade in both the visual and audio over 1s.
        player.play()

        // Wait for SwiftUI to mount the view, then set opacity to 1.
        // The .animation modifier on the view handles the actual animation.
        DispatchQueue.main.asyncAfter(deadline: .now() + 0.2) {
            self.onboardingVideoOpacity = 1.0
            // Fade audio volume from 0 → 1 over 2s to match visual fade
            self.fadeInVideoAudio(player: player, targetVolume: 1.0, duration: 2.0)
        }

        // At 40 seconds into the video, trigger the onboarding demo where
        // Clicky flies to something interesting on screen and comments on it
        let demoTriggerTime = CMTime(seconds: 40, preferredTimescale: 600)
        onboardingDemoTimeObserver = player.addBoundaryTimeObserver(
            forTimes: [NSValue(time: demoTriggerTime)],
            queue: .main
        ) { [weak self] in
            ClickyAnalytics.trackOnboardingDemoTriggered()
            self?.performOnboardingDemoInteraction()
        }

        // Fade out and clean up when the video finishes
        onboardingVideoEndObserver = NotificationCenter.default.addObserver(
            forName: AVPlayerItem.didPlayToEndTimeNotification,
            object: player.currentItem,
            queue: .main
        ) { [weak self] _ in
            guard let self else { return }
            ClickyAnalytics.trackOnboardingVideoCompleted()
            self.onboardingVideoOpacity = 0.0
            // Wait for the 2s fade-out animation to complete before tearing down
            DispatchQueue.main.asyncAfter(deadline: .now() + 2.0) {
                self.tearDownOnboardingVideo()
                // After the video disappears, stream in the prompt to try talking
                DispatchQueue.main.asyncAfter(deadline: .now() + 0.3) {
                    self.startOnboardingPromptStream()
                }
            }
        }
    }

    func tearDownOnboardingVideo() {
        showOnboardingVideo = false
        if let timeObserver = onboardingDemoTimeObserver {
            onboardingVideoPlayer?.removeTimeObserver(timeObserver)
            onboardingDemoTimeObserver = nil
        }
        onboardingVideoPlayer?.pause()
        onboardingVideoPlayer = nil
        if let observer = onboardingVideoEndObserver {
            NotificationCenter.default.removeObserver(observer)
            onboardingVideoEndObserver = nil
        }
    }

    private func startOnboardingPromptStream() {
        let message = "press control + option and introduce yourself"
        onboardingPromptText = ""
        showOnboardingPrompt = true
        onboardingPromptOpacity = 0.0

        withAnimation(.easeIn(duration: 0.4)) {
            onboardingPromptOpacity = 1.0
        }

        var currentIndex = 0
        Timer.scheduledTimer(withTimeInterval: 0.03, repeats: true) { timer in
            guard currentIndex < message.count else {
                timer.invalidate()
                // Auto-dismiss after 10 seconds
                DispatchQueue.main.asyncAfter(deadline: .now() + 10.0) {
                    guard self.showOnboardingPrompt else { return }
                    withAnimation(.easeOut(duration: 0.3)) {
                        self.onboardingPromptOpacity = 0.0
                    }
                    DispatchQueue.main.asyncAfter(deadline: .now() + 0.35) {
                        self.showOnboardingPrompt = false
                        self.onboardingPromptText = ""
                    }
                }
                return
            }
            let index = message.index(message.startIndex, offsetBy: currentIndex)
            self.onboardingPromptText.append(message[index])
            currentIndex += 1
        }
    }

    /// Gradually raises an AVPlayer's volume from its current level to the
    /// target over the specified duration, creating a smooth audio fade-in.
    private func fadeInVideoAudio(player: AVPlayer, targetVolume: Float, duration: Double) {
        let steps = 20
        let stepInterval = duration / Double(steps)
        let volumeIncrement = (targetVolume - player.volume) / Float(steps)
        var stepsRemaining = steps

        Timer.scheduledTimer(withTimeInterval: stepInterval, repeats: true) { timer in
            stepsRemaining -= 1
            player.volume += volumeIncrement

            if stepsRemaining <= 0 {
                timer.invalidate()
                player.volume = targetVolume
            }
        }
    }

    // MARK: - Onboarding Demo Interaction

    private static let onboardingDemoSystemPrompt = """
    you're clicky, a small blue cursor buddy living on the user's screen. you're showing off during onboarding — look at their screen and find ONE specific, concrete thing to point at. pick something with a clear name or identity: a specific app icon (say its name), a specific word or phrase of text you can read, a specific filename, a specific button label, a specific tab title, a specific image you can describe. do NOT point at vague things like "a window" or "some text" — be specific about exactly what you see.

    make a short quirky 3-6 word observation about the specific thing you picked — something fun, playful, or curious that shows you actually read/recognized it. no emojis ever. NEVER quote or repeat text you see on screen — just react to it. keep it to 6 words max, no exceptions.

    CRITICAL COORDINATE RULE: you MUST only pick elements near the CENTER of the screen. your x coordinate must be between 20%-80% of the image width. your y coordinate must be between 20%-80% of the image height. do NOT pick anything in the top 20%, bottom 20%, left 20%, or right 20% of the screen. no menu bar items, no dock icons, no sidebar items, no items near any edge. only things clearly in the middle area of the screen. if the only interesting things are near the edges, pick something boring in the center instead.

    respond with ONLY your short comment followed by the coordinate tag. nothing else. all lowercase.

    format: your comment [POINT:x,y:label]

    the screenshot images are labeled with their pixel dimensions. use those dimensions as the coordinate space. origin (0,0) is top-left. x increases rightward, y increases downward.
    """

    /// Captures a screenshot and asks Claude to find something interesting to
    /// point at, then triggers the buddy's flight animation. Used during
    /// onboarding to demo the pointing feature while the intro video plays.
    func performOnboardingDemoInteraction() {
        // Don't interrupt an active voice response
        guard voiceState == .idle || voiceState == .responding else { return }

        Task {
            do {
                let screenCaptures = try await CompanionScreenCaptureUtility.captureAllScreensAsJPEG()

                // Only send the cursor screen so Claude can't pick something
                // on a different monitor that we can't point at.
                guard let cursorScreenCapture = screenCaptures.first(where: { $0.isCursorScreen }) else {
                    print("🎯 Onboarding demo: no cursor screen found")
                    return
                }

                let dimensionInfo = " (image dimensions: \(cursorScreenCapture.screenshotWidthInPixels)x\(cursorScreenCapture.screenshotHeightInPixels) pixels)"
                let labeledImages = [(data: cursorScreenCapture.imageData, label: cursorScreenCapture.label + dimensionInfo)]

                let (fullResponseText, _) = try await claudeAPI.analyzeImageStreaming(
                    images: labeledImages,
                    systemPrompt: Self.onboardingDemoSystemPrompt,
                    userPrompt: "look around my screen and find something interesting to point at",
                    onTextChunk: { _ in }
                )

                let parseResult = Self.parsePointingCoordinates(from: fullResponseText)

                guard let pointCoordinate = parseResult.coordinate else {
                    print("🎯 Onboarding demo: no element to point at")
                    return
                }

                let screenshotWidth = CGFloat(cursorScreenCapture.screenshotWidthInPixels)
                let screenshotHeight = CGFloat(cursorScreenCapture.screenshotHeightInPixels)
                let displayWidth = CGFloat(cursorScreenCapture.displayWidthInPoints)
                let displayHeight = CGFloat(cursorScreenCapture.displayHeightInPoints)
                let displayFrame = cursorScreenCapture.displayFrame

                let clampedX = max(0, min(pointCoordinate.x, screenshotWidth))
                let clampedY = max(0, min(pointCoordinate.y, screenshotHeight))
                let displayLocalX = clampedX * (displayWidth / screenshotWidth)
                let displayLocalY = clampedY * (displayHeight / screenshotHeight)
                let appKitY = displayHeight - displayLocalY
                let globalLocation = CGPoint(
                    x: displayLocalX + displayFrame.origin.x,
                    y: appKitY + displayFrame.origin.y
                )

                // Set custom bubble text so the pointing animation uses Claude's
                // comment instead of a random phrase
                detectedElementBubbleText = parseResult.spokenText
                detectedElementScreenLocation = globalLocation
                detectedElementDisplayFrame = displayFrame
                print("🎯 Onboarding demo: pointing at \"\(parseResult.elementLabel ?? "element")\" — \"\(parseResult.spokenText)\"")
            } catch {
                print("⚠️ Onboarding demo error: \(error)")
            }
        }
    }
}
