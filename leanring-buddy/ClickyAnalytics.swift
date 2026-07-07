//
//  ClickyAnalytics.swift
//  leanring-buddy
//
//  Analytics stub. PostHog was removed from this fork — no usage data
//  leaves the machine. The full event API surface is kept as no-ops so
//  the many call sites throughout the app compile unchanged, and so a
//  future analytics backend could be dropped back in here.
//

import Foundation

enum ClickyAnalytics {

    // MARK: - Setup

    static func configure() {
        // No-op: analytics removed from this fork.
    }

    // MARK: - App Lifecycle

    /// Fired once on every app launch in applicationDidFinishLaunching.
    static func trackAppOpened() {
        // No-op: analytics removed from this fork.
    }

    // MARK: - Onboarding

    /// User clicked the Start button to begin onboarding for the first time.
    static func trackOnboardingStarted() {
        // No-op: analytics removed from this fork.
    }

    /// User clicked "Watch Onboarding Again" from the panel footer.
    static func trackOnboardingReplayed() {
        // No-op: analytics removed from this fork.
    }

    /// The onboarding video finished playing to the end.
    static func trackOnboardingVideoCompleted() {
        // No-op: analytics removed from this fork.
    }

    /// The 40s onboarding demo interaction where Clicky points at something.
    static func trackOnboardingDemoTriggered() {
        // No-op: analytics removed from this fork.
    }

    // MARK: - Permissions

    /// All three permissions (accessibility, screen recording, mic) are granted.
    static func trackAllPermissionsGranted() {
        // No-op: analytics removed from this fork.
    }

    /// A single permission was granted. Called when polling detects a change.
    static func trackPermissionGranted(permission: String) {
        // No-op: analytics removed from this fork.
    }

    // MARK: - Voice Interaction

    /// User pressed the push-to-talk shortcut (control+option) to start talking.
    static func trackPushToTalkStarted() {
        // No-op: analytics removed from this fork.
    }

    /// User released the shortcut — transcript is being finalized.
    static func trackPushToTalkReleased() {
        // No-op: analytics removed from this fork.
    }

    /// Transcription completed and the user's message is being sent to the AI.
    static func trackUserMessageSent(transcript: String) {
        // No-op: analytics removed from this fork.
    }

    /// Claude responded and the response is being spoken via TTS.
    static func trackAIResponseReceived(response: String) {
        // No-op: analytics removed from this fork.
    }

    /// Claude's response included a [POINT:x,y:label] coordinate tag,
    /// so the buddy is flying to point at a UI element.
    static func trackElementPointed(elementLabel: String?) {
        // No-op: analytics removed from this fork.
    }

    // MARK: - Errors

    /// An error occurred during the AI response pipeline.
    static func trackResponseError(error: String) {
        // No-op: analytics removed from this fork.
    }

    /// An error occurred during TTS playback.
    static func trackTTSError(error: String) {
        // No-op: analytics removed from this fork.
    }
}
