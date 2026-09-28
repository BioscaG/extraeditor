/**
 * Colour and contrast tokens (§13.1).
 *
 * The hard problem for text over footage is that the footage changes. White text is
 * invisible over a bright sky and a black drop shadow looks cheap. So text always
 * gets a **scrim** whose strength is driven by the measured luminance behind it,
 * which the plan supplies per overlay.
 */

export type Palette = {
  text: string;
  textInverse: string;
  accent: string;
  accentAlt: string;
  scrim: string;
};

export const palettes: Record<string, Palette> = {
  /** Neutral default: works on any footage, adds no style of its own. */
  neutral: {
    text: "#FFFFFF",
    textInverse: "#0B0B0C",
    accent: "#FFFFFF",
    accentAlt: "#B9BBC0",
    scrim: "#000000",
  },
  /** Festival: saturated stage-light accents against white text. */
  festival: {
    text: "#FFFFFF",
    textInverse: "#120A1F",
    accent: "#FF3D71",
    accentAlt: "#33E1ED",
    scrim: "#0A0612",
  },
  /** Warm: golden-hour and daytime footage. */
  warm: {
    text: "#FFF8F0",
    textInverse: "#1C1208",
    accent: "#FFB03A",
    accentAlt: "#FF6B4A",
    scrim: "#1A1108",
  },
};

export type PaletteName = keyof typeof palettes;

/** WCAG relative luminance of a hex colour, 0–1. */
export const luminance = (hex: string): number => {
  const v = hex.replace("#", "");
  const channel = (i: number) => {
    const c = parseInt(v.slice(i * 2, i * 2 + 2), 16) / 255;
    return c <= 0.03928 ? c / 12.92 : Math.pow((c + 0.055) / 1.055, 2.4);
  };
  return 0.2126 * channel(0) + 0.7152 * channel(1) + 0.0722 * channel(2);
};

export const contrastRatio = (a: string, b: string): number => {
  const la = luminance(a);
  const lb = luminance(b);
  const [hi, lo] = la > lb ? [la, lb] : [lb, la];
  return (hi + 0.05) / (lo + 0.05);
};

/** Minimum contrast for text over footage. Above WCAG AA, because footage moves. */
export const MIN_CONTRAST = 4.5;

/**
 * Scrim opacity needed for `textColor` to stay legible over a background of the
 * given luminance.
 *
 * Returns 0 when the footage already provides enough contrast — an unnecessary scrim
 * is the thing that makes captions look like a template.
 */
export const scrimOpacity = (backgroundLuminance: number, textColor: string): number => {
  const textLum = luminance(textColor);
  const ratio = textLum > backgroundLuminance
    ? (textLum + 0.05) / (backgroundLuminance + 0.05)
    : (backgroundLuminance + 0.05) / (textLum + 0.05);
  if (ratio >= MIN_CONTRAST) return 0;
  // How far the background must move toward the scrim colour to reach MIN_CONTRAST.
  const needed = textLum > backgroundLuminance
    ? (textLum + 0.05) / MIN_CONTRAST - 0.05
    : MIN_CONTRAST * (textLum + 0.05) - 0.05;
  const span = backgroundLuminance - needed;
  if (Math.abs(span) < 1e-6) return 0.5;
  return Math.min(0.85, Math.max(0, span / Math.max(backgroundLuminance, 1e-6)));
};

/** Text shadow that reads as depth rather than as a drop shadow, sized to the type. */
export const textShadow = (fontSize: number, scrim: string): string =>
  `0 ${(fontSize * 0.02).toFixed(1)}px ${(fontSize * 0.08).toFixed(1)}px ${scrim}66`;
