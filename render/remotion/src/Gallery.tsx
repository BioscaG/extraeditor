/**
 * Gallery composition (§13.4).
 *
 * Renders one component × preset on standard synthetic test plates, at whatever
 * aspect ratio the render is invoked with. Synthetic plates rather than real footage
 * so a gallery diff only ever changes when the *component* changes — a perceptual
 * regression test needs a fixed background (§27).
 */

import React from "react";
import { AbsoluteFill, useCurrentFrame, useVideoConfig } from "remotion";
import { z } from "zod";
import { resolve } from "../../../library/components";
import type { ComponentContext } from "../../../library/components/contract";

export const gallerySchema = z.object({
  componentId: z.string(),
  preset: z.string().nullable().default(null),
  params: z.record(z.string(), z.unknown()).default({}),
  bpm: z.number().default(120),
  durationInFrames: z.number().int().default(30),
  label: z.string().default(""),
});

export type GalleryProps = z.infer<typeof gallerySchema>;

/** A test plate: flat colour with a grid, so motion and edges are legible. */
const Plate: React.FC<{ hue: number; label: string }> = ({ hue, label }) => (
  <AbsoluteFill
    style={{
      background: `linear-gradient(135deg, hsl(${hue} 55% 42%), hsl(${hue + 40} 60% 28%))`,
      display: "flex",
      alignItems: "center",
      justifyContent: "center",
    }}
  >
    <div
      style={{
        position: "absolute",
        inset: 0,
        backgroundImage:
          "repeating-linear-gradient(0deg, rgba(255,255,255,0.14) 0 2px, transparent 2px 80px)," +
          "repeating-linear-gradient(90deg, rgba(255,255,255,0.14) 0 2px, transparent 2px 80px)",
      }}
    />
    <span
      style={{
        fontFamily: "Helvetica, Arial, sans-serif",
        fontSize: 120,
        fontWeight: 900,
        color: "rgba(255,255,255,0.85)",
      }}
    >
      {label}
    </span>
  </AbsoluteFill>
);

export const GalleryComposition: React.FC<GalleryProps> = (props) => {
  const { width, height, fps } = useVideoConfig();
  const frame = useCurrentFrame();
  const entry = resolve(props.componentId);

  if (!entry) {
    return (
      <AbsoluteFill style={{ backgroundColor: "#300", color: "white", fontSize: 48, padding: 60 }}>
        Unknown component: {props.componentId}
      </AbsoluteFill>
    );
  }

  const params = { ...(props.preset ? entry.meta.presets?.[props.preset] : {}), ...props.params };
  const ctx: ComponentContext = {
    frame,
    durationInFrames: props.durationInFrames,
    width,
    height,
    fps,
    bpm: props.bpm,
    backgroundLuminance: 0.35,
  };

  const A = <Plate hue={205} label="A" />;
  const B = <Plate hue={15} label="B" />;

  switch (entry.meta.kind) {
    case "transition":
      return (
        <AbsoluteFill>
          <entry.Component ctx={ctx} params={params} from={A} to={B} />
        </AbsoluteFill>
      );
    case "shot_fx":
    case "finishing":
      return (
        <AbsoluteFill>
          <entry.Component ctx={ctx} params={params}>
            {A}
          </entry.Component>
        </AbsoluteFill>
      );
    default:
      // Text and overlay components draw over a plate.
      return (
        <AbsoluteFill>
          {A}
          <entry.Component ctx={ctx} params={params} />
        </AbsoluteFill>
      );
  }
};

export default GalleryComposition;
