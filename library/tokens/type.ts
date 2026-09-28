/**
 * Typography tokens (§13.1).
 *
 * Sizes are expressed as a fraction of **frame height**, not in pixels, so the same
 * component is correctly proportioned at 1080x1920 and at 540p draft resolution and
 * in every aspect ratio. A modular scale keeps the relationships between sizes
 * consistent, which is most of what makes typography look designed rather than
 * arbitrary.
 */

/** Ratio between adjacent steps. 1.25 (major third) reads as confident, not shouty. */
export const SCALE_RATIO = 1.25;

/** Base size as a fraction of frame height. */
export const BASE = 0.035;

const step = (n: number) => BASE * Math.pow(SCALE_RATIO, n);

export const typeScale = {
  caption: step(1),
  body: step(2),
  lead: step(3),
  title: step(5),
  display: step(7),
  hero: step(9),
} as const;

export type TypeSize = keyof typeof typeScale;

/** Pixel size for a type step at a given frame height. */
export const fontSize = (size: TypeSize, frameHeight: number): number =>
  Math.round(typeScale[size] * frameHeight);

export const families = {
  /** Display: titles, stamps, end cards. Heavy weights, tight tracking. */
  display: '"Archivo Black", "Helvetica Neue", Impact, sans-serif',
  /** Text: captions and lower thirds. Must stay legible small and over motion. */
  text: '"Inter", "Helvetica Neue", Arial, sans-serif',
  /** Mono accent: timecodes, counters, technical stamps. */
  mono: '"JetBrains Mono", "SF Mono", Menlo, monospace',
} as const;

export type FontFamily = keyof typeof families;

/**
 * Tracking (letter-spacing) as a fraction of font size.
 *
 * Large display type needs *negative* tracking to avoid looking loose, while small
 * uppercase text needs positive tracking to stay readable — the opposite of what
 * leaving it at the default does.
 */
export const tracking = {
  display: -0.03,
  title: -0.02,
  body: 0,
  caption: 0.01,
  stamp: 0.12,
} as const;

export const leading = {
  tight: 1.05,
  normal: 1.25,
  loose: 1.5,
} as const;

export const weights = {
  regular: 400,
  medium: 500,
  bold: 700,
  black: 900,
} as const;

export type TextStylePreset = {
  family: FontFamily;
  size: TypeSize;
  weight: keyof typeof weights;
  tracking: keyof typeof tracking;
  leading: keyof typeof leading;
  uppercase?: boolean;
};

export const textStyles: Record<string, TextStylePreset> = {
  caption: { family: "text", size: "lead", weight: "bold", tracking: "body", leading: "tight" },
  title: { family: "display", size: "display", weight: "black", tracking: "display", leading: "tight", uppercase: true },
  stamp: { family: "mono", size: "caption", weight: "medium", tracking: "stamp", leading: "normal", uppercase: true },
  lowerThird: { family: "text", size: "body", weight: "medium", tracking: "caption", leading: "normal" },
};

/** Resolve a preset into concrete CSS for a given frame height. */
export const resolveTextStyle = (
  preset: TextStylePreset,
  frameHeight: number,
): React.CSSProperties => {
  const size = fontSize(preset.size, frameHeight);
  return {
    fontFamily: families[preset.family],
    fontSize: size,
    fontWeight: weights[preset.weight],
    letterSpacing: size * tracking[preset.tracking],
    lineHeight: leading[preset.leading],
    textTransform: preset.uppercase ? "uppercase" : undefined,
  };
};
