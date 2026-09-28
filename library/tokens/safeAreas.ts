/**
 * Safe areas per aspect ratio and platform (§13.1).
 *
 * "Captions under platform UI" is on the pitfalls list (§28) and it is entirely
 * avoidable: on a 9:16 feed the bottom ~20% of the frame is covered by the caption
 * text, account name and action buttons, and the top ~10% by the status bar and
 * progress indicator. Insets are fractions of frame size so they hold at any
 * resolution.
 */

export type Insets = { top: number; right: number; bottom: number; left: number };

export type AspectRatio = "9:16" | "16:9" | "1:1" | "4:5";

/** Generous insets that clear the UI of every major vertical-video platform. */
const VERTICAL_SOCIAL: Insets = { top: 0.1, right: 0.07, bottom: 0.2, left: 0.07 };

export const safeAreas: Record<AspectRatio, Record<string, Insets>> = {
  "9:16": {
    /** Nothing overlaid: only optical margins. */
    none: { top: 0.05, right: 0.05, bottom: 0.05, left: 0.05 },
    social: VERTICAL_SOCIAL,
    /** Right side stays clear for the action rail; bottom for the caption block. */
    reels: { top: 0.09, right: 0.16, bottom: 0.22, left: 0.06 },
    tiktok: { top: 0.1, right: 0.18, bottom: 0.24, left: 0.06 },
  },
  "16:9": {
    none: { top: 0.04, right: 0.04, bottom: 0.04, left: 0.04 },
    /** Broadcast title-safe, which also clears YouTube's control bar. */
    social: { top: 0.06, right: 0.06, bottom: 0.12, left: 0.06 },
  },
  "1:1": {
    none: { top: 0.05, right: 0.05, bottom: 0.05, left: 0.05 },
    social: { top: 0.07, right: 0.07, bottom: 0.14, left: 0.07 },
  },
  "4:5": {
    none: { top: 0.05, right: 0.05, bottom: 0.05, left: 0.05 },
    social: { top: 0.08, right: 0.07, bottom: 0.16, left: 0.07 },
  },
};

export const getInsets = (aspect: AspectRatio, platform = "social"): Insets =>
  safeAreas[aspect]?.[platform] ?? safeAreas[aspect].none;

/** Insets as CSS padding for a given frame size. */
export const insetStyle = (
  aspect: AspectRatio,
  width: number,
  height: number,
  platform = "social",
): React.CSSProperties => {
  const i = getInsets(aspect, platform);
  return {
    paddingTop: height * i.top,
    paddingRight: width * i.right,
    paddingBottom: height * i.bottom,
    paddingLeft: width * i.left,
  };
};

/** The safe rectangle in pixels, for validation (§18.2 checks captions against it). */
export const safeRect = (
  aspect: AspectRatio,
  width: number,
  height: number,
  platform = "social",
) => {
  const i = getInsets(aspect, platform);
  return {
    x: width * i.left,
    y: height * i.top,
    width: width * (1 - i.left - i.right),
    height: height * (1 - i.top - i.bottom),
  };
};

export const aspectOf = (width: number, height: number): AspectRatio => {
  const ratio = width / height;
  if (Math.abs(ratio - 9 / 16) < 0.05) return "9:16";
  if (Math.abs(ratio - 16 / 9) < 0.1) return "16:9";
  if (Math.abs(ratio - 1) < 0.05) return "1:1";
  if (Math.abs(ratio - 4 / 5) < 0.05) return "4:5";
  return ratio < 1 ? "9:16" : "16:9";
};
