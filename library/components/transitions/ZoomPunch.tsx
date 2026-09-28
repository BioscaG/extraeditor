/**
 * transition.zoom_punch — a scale hit through the cut.
 *
 * The outgoing shot punches in (or out) and the incoming shot resolves back to rest.
 * The asymmetry is the point: a fast, eased-out push on the way in and a settle on
 * the way out reads as an impact. A symmetric zoom reads as a slow crossfade.
 *
 * The scale never goes below 1 on either shot, so the frame edge is never exposed.
 */

import React from "react";
import { interpolate, useCurrentFrame } from "remotion";
import { z } from "zod";
import { easing, settle } from "../../tokens/easing";
import { timing } from "../../tokens/timing";
import type { ComponentContext, ComponentMeta } from "../contract";
import { progressOf } from "../contract";

export const params = z.object({
  direction: z.enum(["in", "out"]).default("in"),
  /** Peak scale added at the cut. 0.18 = 118% at the hit. */
  amount: z.number().min(0.02).max(0.6).default(0.18),
  /** Radial blur suggestion, as a fraction of frame size. */
  blur: z.number().min(0).max(1).default(0.25),
});

export type ZoomPunchParams = z.infer<typeof params>;

export const meta: ComponentMeta<typeof params> = {
  id: "transition.zoom_punch",
  version: "1.0.0",
  kind: "transition",
  status: "stable",
  duration: { minFrames: timing.punch.minFrames, maxFrames: 16, default: timing.punch },
  energy: ["high"],
  tags: ["scale", "impact", "beat"],
  beatAnchor: "cut_point",
  motionMatch: "zoom_in",
  sfx: { default: "impact.soft", anchor: "cut_point" },
  params,
  presets: {
    subtle: { amount: 0.08, blur: 0.1 },
    aggressive: { amount: 0.35, blur: 0.5 },
  },
  aspectRatios: ["9:16", "16:9", "1:1", "4:5"],
  author: "human",
  intent:
    "Emphasis on a downbeat, especially into a drop. Cheap-looking if used on every " +
    "cut; reserve it for the beats that actually matter.",
};

type Props = {
  ctx: ComponentContext;
  params?: Partial<ZoomPunchParams>;
  from: React.ReactNode;
  to: React.ReactNode;
};

export const ZoomPunch: React.FC<Props> = ({ ctx, params: raw, from, to }) => {
  const p = params.parse(raw ?? {});
  const frame = useCurrentFrame();
  const progress = progressOf({ ...ctx, frame });
  const sign = p.direction === "in" ? 1 : -1;

  // Outgoing: accelerates into the hit. Incoming: settles back with a slight
  // overshoot, which is what makes it land rather than drift.
  const outScale = 1 + sign * p.amount * interpolate(progress, [0, 1], [0, 1], {
    easing: easing.snap,
    extrapolateRight: "clamp",
  });
  const inScale = 1 + sign * p.amount * (1 - settle(progress, 0.75));

  const blurAmount = Math.sin(progress * Math.PI) * p.blur * 12;
  const opacity = interpolate(progress, [0.35, 0.65], [1, 0], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
  });

  const layer = (scale: number, alpha: number): React.CSSProperties => ({
    position: "absolute",
    inset: 0,
    // Scale is clamped at 1 so the frame edge is never revealed.
    transform: `scale(${Math.max(1, scale).toFixed(4)})`,
    filter: `blur(${blurAmount.toFixed(2)}px)`,
    opacity: alpha,
    willChange: "transform, opacity",
  });

  return (
    <div style={{ position: "absolute", inset: 0, overflow: "hidden" }}>
      <div style={layer(inScale, 1)}>{to}</div>
      <div style={layer(outScale, opacity)}>{from}</div>
    </div>
  );
};

export default ZoomPunch;
