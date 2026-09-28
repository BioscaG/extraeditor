/**
 * shot_fx.impact_shake — a decaying shake on named instants.
 *
 * The difference between a shake that reads as impact and one that reads as a broken
 * gimbal is **decay**: real impact energy dissipates fast, so the first two frames carry
 * almost all of the displacement. A constant-amplitude shake over 15 frames is the single
 * most common way this effect is done badly.
 *
 * Displacement is seeded from the frame number via Remotion's `random`, so it is noisy but
 * **deterministic** — the same frame shakes the same way on every render, which §13.2
 * requires and which is also what makes the gallery's perceptual diff meaningful.
 */

import React from "react";
import { random, useCurrentFrame, useVideoConfig } from "remotion";
import { z } from "zod";
import { timing } from "../../tokens/timing";
import type { ComponentContext, ComponentMeta } from "../contract";

export const params = z.object({
  /** Impact times in seconds, relative to the shot's start. */
  at: z.array(z.number()).default([]),
  /** Peak displacement as a fraction of the frame's short edge. */
  amount: z.number().min(0.002).max(0.08).default(0.014),
  /** How long each impact takes to die away, in beats. */
  decayBeats: z.number().min(0.05).max(0.75).default(0.2),
  /** Add a small rotation. Reads as a heavier hit; too much reads as a glitch. */
  rotate: z.boolean().default(true),
  /** Scale up slightly during the shake, so no frame edge is ever exposed. */
  safetyZoom: z.number().min(1).max(1.12).default(1.03),
});

export type ImpactShakeParams = z.infer<typeof params>;

export const meta: ComponentMeta<typeof params> = {
  id: "shot_fx.impact_shake",
  version: "1.0.0",
  kind: "shot_fx",
  status: "stable",
  duration: { minFrames: 1, maxFrames: 3000, default: timing.hold },
  energy: ["high"],
  tags: ["impact", "shake", "beat", "drop"],
  beatAnchor: "peak",
  sfx: { default: "impact.deep", anchor: "peak" },
  params,
  presets: {
    subtle: { amount: 0.008, decayBeats: 0.15, rotate: false },
    heavy: { amount: 0.035, decayBeats: 0.3, rotate: true },
  },
  aspectRatios: ["9:16", "16:9", "1:1", "4:5"],
  author: "human",
  intent:
    "The single hit a drop lands on, or a hard accent. One per section at most — a shake " +
    "on every downbeat reads as a broken camera, not as energy.",
};

type Props = {
  ctx: ComponentContext;
  params?: Partial<ImpactShakeParams>;
  children: React.ReactNode;
};

export const ImpactShake: React.FC<Props> = ({ ctx, params: raw, children }) => {
  const p = params.parse(raw ?? {});
  const frame = useCurrentFrame();
  const { width, height, fps } = useVideoConfig();
  const t = frame / fps;
  const decay = p.decayBeats * (60 / ctx.bpm);

  // Energy from the most recent impact only; earlier ones have already dissipated.
  let energy = 0;
  for (const impact of p.at) {
    if (impact > t) break;
    const age = t - impact;
    if (age > decay) continue;
    // Exponential decay: the first frames carry nearly all the displacement, which is
    // what makes this read as an impact rather than as a wobble.
    energy = Math.exp(-4.5 * (age / decay));
  }

  if (energy <= 0.001) {
    return <div style={{ position: "absolute", inset: 0 }}>{children}</div>;
  }

  const reach = p.amount * Math.min(width, height) * energy;
  // Seeded on the frame so the noise is deterministic (§13.2) — the same frame always
  // shakes identically, which the gallery's perceptual diff depends on.
  const dx = (random(`shake-x-${frame}`) - 0.5) * 2 * reach;
  const dy = (random(`shake-y-${frame}`) - 0.5) * 2 * reach;
  const rotation = p.rotate ? (random(`shake-r-${frame}`) - 0.5) * 2 * energy * 1.2 : 0;
  const scale = 1 + (p.safetyZoom - 1) * energy;

  return (
    <div
      style={{
        position: "absolute",
        inset: 0,
        transform:
          `translate(${dx.toFixed(2)}px, ${dy.toFixed(2)}px) ` +
          `rotate(${rotation.toFixed(3)}deg) scale(${scale.toFixed(4)})`,
        willChange: "transform",
      }}
    >
      {children}
    </div>
  );
};

export default ImpactShake;
