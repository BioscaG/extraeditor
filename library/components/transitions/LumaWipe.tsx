/**
 * transition.luma_wipe — the incoming shot is revealed through a moving soft edge.
 *
 * This is the component a *discovered convention* can use (§12). When the footage has a
 * habit of ending clips with the lens obscured, the darkness itself is the wipe: cutting
 * on it and wiping through it turns an artefact of how the footage was shot into
 * deliberate punctuation.
 *
 * It is a gradient mask rather than a hard edge, because a hard-edged wipe is the single
 * most dated transition in video and no amount of good timing rescues it.
 */

import React from "react";
import { interpolate, useCurrentFrame } from "remotion";
import { z } from "zod";
import { easing } from "../../tokens/easing";
import { timing } from "../../tokens/timing";
import type { ComponentContext, ComponentMeta } from "../contract";
import { progressOf } from "../contract";

export const params = z.object({
  direction: z.enum(["left", "right", "up", "down"]).default("right"),
  /** Softness of the wipe edge as a fraction of the travel. 0.35 reads as light, not as a line. */
  softness: z.number().min(0.05).max(0.8).default(0.35),
  /** Tint the wipe edge, so it reads as light sweeping across rather than a mask moving. */
  glow: z.number().min(0).max(1).default(0.35),
});

export type LumaWipeParams = z.infer<typeof params>;

export const meta: ComponentMeta<typeof params> = {
  id: "transition.luma_wipe",
  version: "1.0.0",
  kind: "transition",
  status: "stable",
  duration: { minFrames: timing.sweep.minFrames, maxFrames: 24, default: timing.sweep },
  energy: ["medium", "high"],
  tags: ["wipe", "reveal", "convention", "light"],
  beatAnchor: "cut_point",
  motionMatch: "horizontal",
  sfx: { default: "whoosh.slow", anchor: "cut_point" },
  params,
  presets: {
    soft: { softness: 0.5, glow: 0.2 },
    hard: { softness: 0.12, glow: 0.5 },
  },
  aspectRatios: ["9:16", "16:9", "1:1", "4:5"],
  author: "human",
  intent:
    "Section changes, and especially cuts on a discovered occlusion motif — where the " +
    "footage already goes dark, a wipe through that darkness reads as intent rather than " +
    "as a mistake being covered up. Not for ordinary cuts: a wipe between two unrelated " +
    "shots is the most dated transition in video.",
};

type Props = {
  ctx: ComponentContext;
  params?: Partial<LumaWipeParams>;
  from: React.ReactNode;
  to: React.ReactNode;
};

export const LumaWipe: React.FC<Props> = ({ ctx, params: raw, from, to }) => {
  const p = params.parse(raw ?? {});
  const frame = useCurrentFrame();
  const progress = progressOf({ ...ctx, frame });

  const angle = { right: 90, left: 270, down: 180, up: 0 }[p.direction];
  // The mask edge travels from before the frame to past it, so the wipe both starts and
  // finishes fully off-screen and never leaves a sliver of the outgoing shot behind.
  const soft = p.softness * 100;
  const position = interpolate(progress, [0, 1], [-soft, 100 + soft], {
    easing: easing.glide,
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
  });

  const mask =
    `linear-gradient(${angle}deg, ` +
    `rgba(0,0,0,1) ${(position - soft).toFixed(2)}%, ` +
    `rgba(0,0,0,0) ${(position + soft).toFixed(2)}%)`;

  // The glow sits on the edge itself, which is what makes it read as light passing over
  // the picture rather than as one layer sliding out from under another.
  const glowGradient =
    `linear-gradient(${angle}deg, ` +
    `rgba(255,255,255,0) ${(position - soft * 1.2).toFixed(2)}%, ` +
    `rgba(255,255,255,${p.glow.toFixed(2)}) ${position.toFixed(2)}%, ` +
    `rgba(255,255,255,0) ${(position + soft * 1.2).toFixed(2)}%)`;

  return (
    <div style={{ position: "absolute", inset: 0, overflow: "hidden" }}>
      <div style={{ position: "absolute", inset: 0 }}>{to}</div>
      <div
        style={{
          position: "absolute",
          inset: 0,
          maskImage: mask,
          WebkitMaskImage: mask,
          willChange: "mask-image",
        }}
      >
        {from}
      </div>
      {p.glow > 0.01 ? (
        <div style={{ position: "absolute", inset: 0, background: glowGradient,
                      mixBlendMode: "screen" }} />
      ) : null}
    </div>
  );
};

export default LumaWipe;
