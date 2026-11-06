#!/usr/bin/env python3
"""Stream a USB capture card to a window.

Press the spacebar to snapshot the current frame, send it to Gemini, and show
the answer as an overlay on the output window.
"""

import argparse
import os
import threading

import cv2

from google import genai
from google.genai import types

# Model requested for the answer.
MODEL = "gemini-flash-lite-latest"

# Shared GUI state updated by the background request thread.
_overlay_state = {"pending": False, "response": ""}
_overlay_lock = threading.Lock()

_PROMPT = (
    "For each question visible in the image, give a concise walkthrough: the "
    "key steps taken and the final answer.\n"
    "No long explanations, no preamble, no filler. Just the steps and the "
    "answer. If several questions are visible, address them in order."
)


def _wrap(text, max_chars):
    """Wrap free text into lines no longer than max_chars characters."""
    words = text.split()
    lines = []
    current = ""
    for word in words:
        candidate = word if not current else f"{current} {word}"
        if len(candidate) <= max_chars:
            current = candidate
        else:
            if current:
                lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines or [""]


def _draw_overlay(img, text):
    """Draw the given text inside a translucent box at the bottom of the image."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.55
    thickness = 1
    max_chars = 64

    lines = _wrap(text, max_chars)
    height, width = img.shape[:2]
    text_size = cv2.getTextSize("Ag", font, scale, thickness)
    text_w, text_h = text_size[0]
    line_h = int(text_h * 1.8)
    per_char = text_w / 2.0

    box_w = min(int(per_char * max_chars) + 20, width - 10)
    box_h = len(lines) * line_h + 16
    x0, y0 = 10, height - box_h - 10
    x1, y1 = x0 + box_w, y0 + box_h

    # Translucent black box behind the text.
    overlay = img.copy()
    cv2.rectangle(overlay, (x0, y0), (x1, y1), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, img, 0.4, 0, img)

    # White text, one line per row.
    for i, line in enumerate(lines):
        y = y0 + 14 + (i + 1) * line_h
        cv2.putText(img, line, (x0 + 8, y), font, scale, (255, 255, 255),
                    thickness, cv2.LINE_AA)
    return img


def _ask_gemini(client, img_bytes):
    """Send a captured frame to Gemini and store the answer in the overlay state."""
    try:
        response = client.models.generate_content(
            model=MODEL,
            contents=[
                types.Part.from_text(text=_PROMPT),
                types.Part.from_bytes(data=img_bytes, mime_type="image/png"),
            ],
        )
        answer = (response.text or "").strip()
    except Exception as exc:  # network, auth, or API error -> show it as the answer
        answer = f"(Error contacting Gemini: {exc})"

    with _overlay_lock:
        _overlay_state["pending"] = False
        _overlay_state["response"] = answer


def main():
    parser = argparse.ArgumentParser(description="Stream a USB capture card to a window.")
    parser.add_argument(
        "--device",
        default=0,
        type=int,
        help="Camera index or video path (default: 0)",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=0,
        help="Requested capture width in pixels (0 = leave default)",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=0,
        help="Requested capture height in pixels (0 = leave default)",
    )
    parser.add_argument(
        "--backend",
        choices=["CAP_ANY", "CAP_V4L2", "CAP_MSMF", "CAP_AVFOUNDATION"],
        default="CAP_ANY",
        help="OpenCV capture backend (default: CAP_ANY)",
    )
    args = parser.parse_args()

    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        print("Error: no Gemini API key found.")
        print("       Set one with:  export GEMINI_API_KEY='...'")
        return 1

    backend = getattr(cv2, args.backend)
    client = genai.Client(api_key=api_key)
    cap = cv2.VideoCapture(args.device, backend)

    if args.width:
        cap.set(cv2.CAP_PROP_WIDTH, float(args.width))
    if args.height:
        cap.set(cv2.CAP_PROP_HEIGHT, float(args.height))

    if not cap.isOpened():
        print(f"Error: could not open device {args.device}")
        return 1

    # Grab a first frame so the window resizes to the actual resolution.
    ok, frame = cap.read()
    if not ok:
        print("Error: could not read a frame from the capture device")
        cap.release()
        return 1

    height, width = frame.shape[:2]
    window_name = "USB Capture"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, width, height)

    print("Press 'q' to quit. Press SPACE to ask Gemini for the answer.")
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Error: failed to read frame, ending stream")
                break

            with _overlay_lock:
                pending = _overlay_state["pending"]
                response = _overlay_state["response"]

            if pending:
                display = _draw_overlay(frame.copy(), "Requesting answer…")
            elif response:
                display = _draw_overlay(frame.copy(), response)
            else:
                display = frame
            cv2.imshow(window_name, display)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == 32 and not pending:  # spacebar
                img_bytes = cv2.imencode(".png", frame)[1].tobytes()
                with _overlay_lock:
                    _overlay_state["pending"] = True
                    _overlay_state["response"] = ""
                threading.Thread(
                    target=_ask_gemini,
                    args=(client, img_bytes),
                    daemon=True,
                ).start()
    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
