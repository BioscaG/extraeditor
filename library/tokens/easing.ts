/**
 * Easing tokens (§13.1).
 *
 * A small curated set. Linear motion is the single clearest tell of an amateur
 * edit — nothing in the physical world moves linearly — so it is deliberately not
 * exported as a named token.
 *
 * Each easing has one job:
 *  - `snap`   an exponential-out: leaves instantly, arrives softly. Impacts, cuts.
 *  - `glide`  a cubic in-out: symmetric and calm. Reframes, slow pushes.
 *  - `settle` a damped spring: overshoots slightly then settles. Text, punches.
 *  - `ease`   a gentle cubic-out for everything else.
 */

import { Easing } from "remotion";

export const easing = {
  snap: Easing.out(Easing.exp),
  glide: Easing.inOut(Easing.cubic),
  ease: Easing.out(Easing.cubic),
  /** Anticipation: pulls back slightly before moving. Use sparingly. */
  anticipate: Easing.inOut(Easing.back(1.4)),
} as const;

export type EasingToken = keyof typeof easing;

/**
 * A critically-damped spring, as a pure function of progress.
 *
 * Remotion's `spring()` needs frame and fps; this is the closed form so components
 * can use it inside plain `interpolate` calls and stay frame-deterministic (§13.2).
 * `damping` above ~0.8 barely overshoots; below ~0.5 it bounces visibly.
 */
export const settle = (progress: number, damping = 0.7): number => {
  if (progress <= 0) return 0;
  if (progress >= 1) return 1;
  const omega = 8 + 12 * (1 - damping);
  return 1 - Math.exp(-damping * omega * progress) * Math.cos(omega * (1 - damping) * progress);
};

export const getEasing = (token: EasingToken | undefined) =>
  token ? easing[token] : easing.ease;
