/**
 * transition.whip_pan — a directional whip with motion blur.
 *
 * The outgoing shot accelerates off one edge while the incoming shot arrives from
 * the other, both smeared by directional blur. It reads as a camera whip, so it only
 * looks right when the footage is already moving that way — hence
 * `motionMatch: "horizontal"`, which the director uses to pick shots whose measured
 * motion direction agrees (§9 `motion`).
 *
 * The blur is the whole effect. Without it this is a slide, which looks like a
 * presentation template.
 */

import React from "react";
import { interpolate, useCurrentFrame, useVideoConfig } from "remotion";
import { z } from "zod";
import { easing } from "../../tokens/easing";
import { timing } from "../../tokens/timing";
import type { ComponentContext, ComponentMeta } from "../contract";
import { progressOf } from "../contract";

export const params = z.object({
  direction: z.enum(["left", "right", "up", "down"]).default("left"),
  /** Peak blur along the motion axis, as a fraction of the frame's short edge. */
  blur: z.number().min(0).max(1).default(0.6),
  /**
   * Peak stretch along the motion axis at the midpoint. 1.12 = 112%.
   *
   * This is the signature of a real whip: the image smears *and* stretches. An
   * earlier version instead over-travelled past the frame edge, which separated the
   * two shots and exposed a black band through the middle of the transition.
   */
  stretch: z.number().min(1).max(1.6).default(1.12),
});

export type WhipPanParams = z.infer<typeof params>;

export const meta: ComponentMeta<typeof params> = {
  id: "transition.whip_pan",
  version: "1.0.0",
  kind: "transition",
  status: "stable",
  duration: { minFrames: timing.sweep.minFrames, maxFrames: 20, default: timing.sweep },
  energy: ["high"],
  tags: ["movement", "directional", "camera"],
  beatAnchor: "cut_point",
  motionMatch: "horizontal",
  sfx: { default: "whoosh.fast", anchor: "cut_point" },
  params,
  presets: {
    subtle: { blur: 0.3, stretch: 1.05 },
    aggressive: { blur: 0.8, stretch: 1.25 },
  },
  aspectRatios: ["9:16", "16:9", "1:1", "4:5"],
  author: "human",
  intent:
    "Section changes and high-energy runs, where both shots already move in the " +
    "same direction. Not between static shots: with nothing moving it reads as a slide.",
};

type Props = {
  ctx: ComponentContext;
  params?: Partial<WhipPanParams>;
  /** The outgoing shot. */
  from: React.ReactNode;
  /** The incoming shot. */
  to: React.ReactNode;
};

export const WhipPan: React.FC<Props> = ({ ctx, params: raw, from, to }) => {
  const p = params.parse(raw ?? {});
  const frame = useCurrentFrame();
  const { width, height } = useVideoConfig();
  const progress = progressOf({ ...ctx, frame });

  const horizontal = p.direction === "left" || p.direction === "right";
  const span = horizontal ? width : height;
  const sign = p.direction === "left" || p.direction === "up" ? -1 : 1;

  // Both shots share one eased position so they move as a single gesture, and they
  // sit exactly one frame-span apart so the pair always covers the frame.
  const travel = interpolate(progress, [0, 1], [0, span * sign], {
    easing: easing.snap,
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
  });

  // Blur and stretch both peak at the midpoint — where the cut is — and vanish at
  // each end, so the shots either side of the transition are sharp and undistorted.
  const peak = Math.sin(progress * Math.PI);
  const blurPx = peak * p.blur * (Math.min(width, height) / 45);
  const stretch = 1 + (p.stretch - 1) * peak;

  // Directional blur: CSS `blur()` is isotropic, which reads as "out of focus"
  // rather than "moving fast". An SVG feGaussianBlur takes a per-axis stdDeviation.
  const filterId = `whip-${p.direction}`;
  const stdDeviation = horizontal ? `${blurPx.toFixed(2)} 0` : `0 ${blurPx.toFixed(2)}`;

  const layer = (value: number): React.CSSProperties => ({
    position: "absolute",
    inset: 0,
    transform: horizontal
      ? `translateX(${value.toFixed(2)}px) scaleX(${stretch.toFixed(4)})`
      : `translateY(${value.toFixed(2)}px) scaleY(${stretch.toFixed(4)})`,
    filter: blurPx > 0.05 ? `url(#${filterId})` : undefined,
    willChange: "transform, filter",
  });

  return (
    <div style={{ position: "absolute", inset: 0, overflow: "hidden" }}>
      <svg width={0} height={0} style={{ position: "absolute" }} aria-hidden>
        <defs>
          <filter id={filterId} x="-20%" y="-20%" width="140%" height="140%">
            <feGaussianBlur stdDeviation={stdDeviation} edgeMode="duplicate" />
          </filter>
        </defs>
      </svg>
      <div style={layer(travel)}>{from}</div>
      <div style={layer(travel - span * sign)}>{to}</div>
    </div>
  );
};

export default WhipPan;
