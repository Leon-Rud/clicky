//
//  AgentActionExecutor.swift
//  leanring-buddy
//
//  Synthesizes keyboard typing, keyboard shortcuts, and scrolling via CGEvent
//  for multi-step agent tasks. Mouse clicks stay in CompanionManager (they
//  are coupled to the overlay flight animation); everything else the agent
//  loop can do to the machine lives here.
//

import AppKit
import CoreGraphics
import Foundation

enum AgentActionExecutor {

    // MARK: - Typing

    /// Types arbitrary text into whatever UI element currently has keyboard
    /// focus, using CGEvent's unicode-string payload (works for any character,
    /// independent of the user's keyboard layout).
    ///
    /// The text is typed in small chunks with a short pause between them —
    /// some apps drop characters when a single event carries a long unicode
    /// string or when events arrive with no gap.
    static func typeText(_ textToType: String) {
        let maximumCharactersPerEvent = 16
        var remainingCharacters = Substring(textToType)

        while !remainingCharacters.isEmpty {
            let chunk = remainingCharacters.prefix(maximumCharactersPerEvent)
            remainingCharacters = remainingCharacters.dropFirst(chunk.count)

            let chunkUTF16CodeUnits = Array(String(chunk).utf16)
            guard let keyDownEvent = CGEvent(keyboardEventSource: nil, virtualKey: 0, keyDown: true),
                  let keyUpEvent = CGEvent(keyboardEventSource: nil, virtualKey: 0, keyDown: false) else {
                continue
            }
            keyDownEvent.keyboardSetUnicodeString(
                stringLength: chunkUTF16CodeUnits.count,
                unicodeString: chunkUTF16CodeUnits
            )
            keyUpEvent.keyboardSetUnicodeString(
                stringLength: chunkUTF16CodeUnits.count,
                unicodeString: chunkUTF16CodeUnits
            )
            keyDownEvent.post(tap: .cghidEventTap)
            keyUpEvent.post(tap: .cghidEventTap)

            // 15ms between chunks keeps fast typing while letting the target
            // app's input pipeline keep up.
            usleep(15_000)
        }
        print("⌨️ Agent action: typed \(textToType.count) characters")
    }

    // MARK: - Keyboard shortcuts

    /// Presses a single key or a modifier combo described in plain text, like
    /// "return", "escape", "cmd+t", or "cmd+shift+t". Returns false when the
    /// description contains a key name this executor doesn't know.
    static func pressKeyCombo(_ keyComboDescription: String) -> Bool {
        var modifierFlags: CGEventFlags = []
        var baseKeyName: String? = nil

        for comboPart in keyComboDescription.lowercased().split(separator: "+") {
            let partName = comboPart.trimmingCharacters(in: .whitespaces)
            switch partName {
            case "cmd", "command":
                modifierFlags.insert(.maskCommand)
            case "shift":
                modifierFlags.insert(.maskShift)
            case "opt", "option", "alt":
                modifierFlags.insert(.maskAlternate)
            case "ctrl", "control":
                modifierFlags.insert(.maskControl)
            default:
                baseKeyName = partName
            }
        }

        guard let baseKeyName,
              let virtualKeyCode = Self.virtualKeyCode(forKeyName: baseKeyName),
              let keyDownEvent = CGEvent(keyboardEventSource: nil, virtualKey: virtualKeyCode, keyDown: true),
              let keyUpEvent = CGEvent(keyboardEventSource: nil, virtualKey: virtualKeyCode, keyDown: false) else {
            print("⚠️ Agent action: unknown key combo \"\(keyComboDescription)\"")
            return false
        }

        keyDownEvent.flags = modifierFlags
        keyUpEvent.flags = modifierFlags
        keyDownEvent.post(tap: .cghidEventTap)
        keyUpEvent.post(tap: .cghidEventTap)

        print("⌨️ Agent action: pressed \"\(keyComboDescription)\"")
        return true
    }

    // MARK: - Scrolling

    enum ScrollDirection {
        case up
        case down
    }

    /// Scrolls the content under the given point (global CGEvent coordinates,
    /// top-left origin on the primary screen). Scroll events are delivered to
    /// the window under the pointer, so the real cursor is warped to the
    /// target point for the scroll and restored afterwards — same pattern as
    /// CompanionManager's synthesized clicks.
    static func scroll(
        atGlobalCGEventPoint scrollPointInCGEventCoordinates: CGPoint,
        direction: ScrollDirection,
        lineCount: Int = 8
    ) {
        let originalCursorPositionInCGEventCoordinates = CGEvent(source: nil)?.location

        CGWarpMouseCursorPosition(scrollPointInCGEventCoordinates)

        // Positive wheel values scroll up (content moves down), negative down.
        let signedLineCount = Int32(direction == .up ? lineCount : -lineCount)
        if let scrollEvent = CGEvent(
            scrollWheelEvent2Source: nil,
            units: .line,
            wheelCount: 1,
            wheel1: signedLineCount,
            wheel2: 0,
            wheel3: 0
        ) {
            scrollEvent.location = scrollPointInCGEventCoordinates
            scrollEvent.post(tap: .cghidEventTap)
        }

        if let originalCursorPositionInCGEventCoordinates {
            CGWarpMouseCursorPosition(originalCursorPositionInCGEventCoordinates)
        }

        print("🖱️ Agent action: scrolled \(direction == .up ? "up" : "down") \(lineCount) lines")
    }

    // MARK: - Virtual key codes

    /// ANSI-layout virtual key codes (from Carbon's Events.h). Covers letters,
    /// digits, and every named key the agent-step prompt advertises.
    private static func virtualKeyCode(forKeyName keyName: String) -> CGKeyCode? {
        let namedKeyCodes: [String: CGKeyCode] = [
            "a": 0, "s": 1, "d": 2, "f": 3, "h": 4, "g": 5, "z": 6, "x": 7,
            "c": 8, "v": 9, "b": 11, "q": 12, "w": 13, "e": 14, "r": 15,
            "y": 16, "t": 17, "o": 31, "u": 32, "i": 34, "p": 35, "l": 37,
            "j": 38, "k": 40, "n": 45, "m": 46,
            "1": 18, "2": 19, "3": 20, "4": 21, "5": 23, "6": 22, "7": 26,
            "8": 28, "9": 25, "0": 29,
            "return": 36, "enter": 36, "tab": 48, "space": 49, "spacebar": 49,
            "delete": 51, "backspace": 51, "escape": 53, "esc": 53,
            "left": 123, "right": 124, "down": 125, "up": 126,
            "home": 115, "end": 119, "pageup": 116, "pagedown": 121,
            "forwarddelete": 117,
            "minus": 27, "-": 27, "equals": 24, "=": 24, "comma": 43, ",": 43,
            "period": 47, ".": 47, "slash": 44, "/": 44, "semicolon": 41,
            "quote": 39, "backslash": 42, "grave": 50, "`": 50,
            "[": 33, "]": 30,
        ]
        return namedKeyCodes[keyName]
    }
}
