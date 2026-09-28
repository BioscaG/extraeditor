/**
 * shot_fx.beat_pulse — a subtle scale pulse on each downbeat.
 *
 * Deliberately small. A pulse you *notice* is too big; the effect should register as
 * the picture breathing with the track. The default 2.5% is near the threshold of
 * perception, which is exactly where it belongs.
 *
 * Pulses are placed on absolute beat times passed in by the plan rather than
 * generated from a period, so they stay locked to the fitted beat grid even where the
 * music was edited and the phase shifted.
 */

import React from "react";
import { useCurrentFrame, useVideoConfig } from "remotion";
import { z } from "zod";
import { settle } from "../../tokens/easing";
import { timing } from "../../tokens/timing";
import type { ComponentContext, ComponentMeta } from "../contract";

export const params = z.object({
  /** Beat times in seconds, relative to the shot's start on the timeline. */
  beats: z.array(z.number()).default([]),
  /** Peak scale added on the hit. 0.025 = 102.5%. */
  amount: z.number().min(0.005).max(0.15).default(0.025),
  /** How long each pulse takes to settle back, in beats. */
  decayBeats: z.number().min(0.1).max(1).default(0.4),
});

export type BeatPulseParams = z.infer<typeof params>;

export const meta: ComponentMeta<typeof params> = {
  id: "shot_fx.beat_pulse",
  version: "1.0.0",
  kind: "shot_fx",
  status: "stable",
  duration: { minFrames: 1, maxFrames: 3000, default: timing.hold },
  energy: ["medium", "high"],
  tags: ["beat", "scale", "subtle"],
  beatAnchor: "peak",
  params,
  presets: {
    barely: { amount: 0.012, decayBeats: 0.3 },
    visible: { amount: 0.06, decayBeats: 0.5 },
  },
  aspectRatios: ["9:16", "16:9", "1:1", "4:5"],
  author: "human",
  intent:
    "Long shots that would otherwise feel static under a driving track. If the pulse " +
    "is noticeable it is too strong.",
};

type Props = {
  ctx: ComponentContext;
  params?: Partial<BeatPulseParams>;
  children: React.ReactNode;
};

export const BeatPulse: React.FC<Props> = ({ ctx, params: raw, children }) => {
  const p = params.parse(raw ?? {});
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();
  const t = frame / fps;
  const decay = p.decayBeats * (60 / ctx.bpm);

  // The most recent beat at or before now drives the pulse; earlier ones have decayed.
  let scale = 1;
  for (const beat of p.beats) {
    if (beat > t) break;
    const age = t - beat;
    if (age > decay) continue;
    // settle() rises to 1; invert it so the pulse starts at full and relaxes.
    scale = 1 + p.amount * (1 - settle(age / decay, 0.85));
  }

  return (
    <div
      style={{
        position: "absolute",
        inset: 0,
        transform: `scale(${scale.toFixed(5)})`,
        willChange: "transform",
      }}
    >
      {children}
    </div>
  );
};

export default BeatPulse;
