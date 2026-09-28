/**
 * finishing.finish — grain, vignette and halation over the whole frame.
 *
 * Why this exists in Remotion at all, when ffmpeg can do grain and vignette: **halation**.
 * The bloom around highlights is what makes digital footage read as filmed rather than
 * recorded, and doing it properly needs a blurred, threshold-masked copy of the frame
 * composited back over itself — which is a compositing operation, not a filter.
 *
 * Grain and vignette are here too so the whole finishing pass is one layer with one set of
 * parameters, rather than split across two stages that have to be kept consistent.
 *
 * Everything here is deliberately *subtle*. Finishing you can see is finishing that has
 * been overdone; its job is to make the picture cohere, not to be noticed.
 */

import React from "react";
import { random, useCurrentFrame, useVideoConfig } from "remotion";
import { z } from "zod";
import { timing } from "../../tokens/timing";
import type { ComponentContext, ComponentMeta } from "../contract";

export const params = z.object({
  /** Grain intensity. Above ~0.25 it stops looking like film and starts looking noisy. */
  grain: z.number().min(0).max(0.5).default(0.12),
  /** Vignette strength. */
  vignette: z.number().min(0).max(0.7).default(0.22),
  /** Halation: bloom around highlights. The reason this component is not an ffmpeg filter. */
  halation: z.number().min(0).max(0.8).default(0.28),
  /** Warmth of the halation, 0 = white bloom, 1 = orange. Film halation is warm. */
  halationWarmth: z.number().min(0).max(1).default(0.6),
});

export type FinishParams = z.infer<typeof params>;

export const meta: ComponentMeta<typeof params> = {
  id: "finishing.finish",
  version: "1.0.0",
  kind: "finishing",
  status: "stable",
  duration: { minFrames: 1, maxFrames: 30000, default: timing.hold },
  energy: ["low", "medium", "high"],
  tags: ["grain", "vignette", "halation", "film"],
  params,
  presets: {
    clean: { grain: 0.05, vignette: 0.12, halation: 0.12 },
    filmic: { grain: 0.18, vignette: 0.28, halation: 0.4 },
    none: { grain: 0, vignette: 0, halation: 0 },
  },
  aspectRatios: ["9:16", "16:9", "1:1", "4:5"],
  author: "human",
  intent:
    "Applied once over the finished picture to make shots from different phones cohere. " +
    "Keep it below the threshold of notice: visible grain and a visible vignette are the " +
    "two clearest signs of a first-time colourist.",
};

type Props = {
  ctx: ComponentContext;
  params?: Partial<FinishParams>;
  children: React.ReactNode;
};

export const Finish: React.FC<Props> = ({ params: raw, children }) => {
  const p = params.parse(raw ?? {});
  const frame = useCurrentFrame();
  const { width, height } = useVideoConfig();
  const short = Math.min(width, height);

  // Grain is regenerated per frame from a seeded source, so it moves like film grain
  // rather than sitting still like a texture — but deterministically (§13.2).
  const grainSeed = Math.floor(random(`grain-${frame}`) * 1000);
  const warm = `rgba(255, ${Math.round(255 - 60 * p.halationWarmth)}, ` +
    `${Math.round(255 - 130 * p.halationWarmth)}, `;

  return (
    <div style={{ position: "absolute", inset: 0, overflow: "hidden" }}>
      <div style={{ position: "absolute", inset: 0 }}>{children}</div>

      {p.halation > 0.01 ? (
        // A blurred, brightness-thresholded copy screened back over the picture: only the
        // highlights survive the threshold, so only they bloom.
        <div
          style={{
            position: "absolute",
            inset: 0,
            filter: `blur(${(short * 0.018).toFixed(1)}px) brightness(1.5) contrast(2.6) saturate(0.7)`,
            mixBlendMode: "screen",
            opacity: p.halation * 0.5,
            pointerEvents: "none",
          }}
        >
          {children}
        </div>
      ) : null}

      {p.vignette > 0.01 ? (
        <div
          style={{
            position: "absolute",
            inset: 0,
            background:
              `radial-gradient(ellipse at center, rgba(0,0,0,0) 45%, ` +
              `rgba(0,0,0,${(p.vignette * 0.85).toFixed(3)}) 100%)`,
            pointerEvents: "none",
          }}
        />
      ) : null}

      {p.grain > 0.005 ? (
        <svg
          style={{
            position: "absolute",
            inset: 0,
            width: "100%",
            height: "100%",
            opacity: p.grain,
            mixBlendMode: "overlay",
            pointerEvents: "none",
          }}
          aria-hidden
        >
          <filter id={`grain-${frame}`}>
            <feTurbulence
              type="fractalNoise"
              baseFrequency="0.8"
              numOctaves={2}
              seed={grainSeed}
              stitchTiles="stitch"
            />
            <feColorMatrix type="saturate" values="0" />
          </filter>
          <rect width="100%" height="100%" filter={`url(#grain-${frame})`} />
        </svg>
      ) : null}

      {p.halation > 0.01 ? (
        <div
          style={{
            position: "absolute",
            inset: 0,
            background: `radial-gradient(ellipse at center, ${warm}${(p.halation * 0.07).toFixed(3)}) 0%, ${warm}0) 70%)`,
            mixBlendMode: "screen",
            pointerEvents: "none",
          }}
        />
      ) : null}
    </div>
  );
};

export default Finish;
