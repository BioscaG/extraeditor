/**
 * transition.dip — a dip through white or black, or a single flash frame.
 *
 * The cheapest designed transition there is, and the one that survives overuse best,
 * because it reads as punctuation rather than as an effect. Three variants in one
 * component because they are the same gesture at different durations: a 2-frame dip is
 * a flash, a 10-frame dip is a breath.
 *
 * The curve matters more than the colour. A linear fade to white looks like a crossfade
 * with a bug; `snap` in and out makes it land.
 */

import React from "react";
import { interpolate, useCurrentFrame } from "remotion";
import { z } from "zod";
import { easing } from "../../tokens/easing";
import { timing } from "../../tokens/timing";
import type { ComponentContext, ComponentMeta } from "../contract";
import { progressOf } from "../contract";

export const params = z.object({
  color: z.enum(["white", "black"]).default("white"),
  /** Peak opacity of the dip. Below 1 the shots stay faintly visible through it. */
  strength: z.number().min(0.2).max(1).default(1),
  /** How much of the duration is spent at full dip. 0 = instant crossover. */
  hold: z.number().min(0).max(0.6).default(0.1),
});

export type DipParams = z.infer<typeof params>;

export const meta: ComponentMeta<typeof params> = {
  id: "transition.dip",
  version: "1.0.0",
  kind: "transition",
  status: "stable",
  duration: { minFrames: 2, maxFrames: 24, default: timing.punch },
  energy: ["medium", "high"],
  tags: ["dip", "flash", "punctuation", "invisible"],
  beatAnchor: "cut_point",
  sfx: { default: "impact.soft", anchor: "cut_point" },
  params,
  presets: {
    flash: { color: "white", strength: 1, hold: 0 },
    breath: { color: "black", strength: 1, hold: 0.25 },
    soft: { color: "white", strength: 0.6, hold: 0.1 },
  },
  aspectRatios: ["9:16", "16:9", "1:1", "4:5"],
  author: "human",
  intent:
    "Punctuation between sections, or a hit on a downbeat. A white flash reads as energy " +
    "and a black dip as a breath. The one designed transition that survives repetition — " +
    "but only because it is almost invisible, so it cannot carry a section change alone.",
};

type Props = {
  ctx: ComponentContext;
  params?: Partial<DipParams>;
  from: React.ReactNode;
  to: React.ReactNode;
};

export const Dip: React.FC<Props> = ({ ctx, params: raw, from, to }) => {
  const p = params.parse(raw ?? {});
  const frame = useCurrentFrame();
  const progress = progressOf({ ...ctx, frame });

  // Dip in, hold, dip out. The hold is what separates a flash from a breath.
  //
  // `hold: 0` is a legal value (the `flash` preset uses it) and would put two identical
  // stops at 0.5, which `interpolate` rejects — its input range must be strictly
  // increasing. So the stops are built to be strictly increasing whatever `hold` is: at
  // zero hold the curve is simply up-then-down through a single peak.
  const hold = Math.max(0, Math.min(0.6, p.hold));
  const stops = hold <= 1e-6
    ? [0, 0.5, 1]
    : [0, (1 - hold) / 2, 1 - (1 - hold) / 2, 1];
  const values = hold <= 1e-6
    ? [0, p.strength, 0]
    : [0, p.strength, p.strength, 0];
  const cover = interpolate(progress, stops, values, {
    easing: easing.snap,
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
  });

  // The shots swap under the cover, at its peak, so the cut itself is never seen.
  const showIncoming = progress >= 0.5;

  return (
    <div style={{ position: "absolute", inset: 0, overflow: "hidden" }}>
      <div style={{ position: "absolute", inset: 0 }}>{showIncoming ? to : from}</div>
      <div
        style={{
          position: "absolute",
          inset: 0,
          backgroundColor: p.color === "white" ? "#FFFFFF" : "#000000",
          opacity: cover,
          willChange: "opacity",
        }}
      />
    </div>
  );
};

export default Dip;
